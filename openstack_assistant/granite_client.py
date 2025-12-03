"""Granite API client for the OpenStack Upgrade Assistant."""

import ast
import asyncio
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from transformers import AutoTokenizer

from .config import Config
from .tools import get_openstack_tools

logger = logging.getLogger(__name__)
console = Console()


class GraniteClient:
    """Client for interacting with Granite LLM API.

    Attributes:
        config: Configuration object
        session: Requests session for API calls
        messages: Conversation history (includes system instruction as first message if provided)
        tools: Available MCP tools
        mcp_client: MCP client for tool execution
    """

    def __init__(self, config: Config):
        """Initialize the Granite client.

        Args:
            config: Configuration object containing API URL and User Key
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.granite_user_key}",
            "Content-Type": "application/json"
        })

        # Disable SSL verification for internal/development endpoints with self-signed certs
        # This is needed for some internal Red Hat API endpoints
        self.session.verify = False

        # Suppress only the single InsecureRequestWarning from urllib3
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.messages = []
        self.tools = None
        self.mcp_client = None
        self.last_usage_metadata = None  # Store usage metadata from last response

        # Initialize tokenizer for chat template formatting
        model_id = "ibm-granite/granite-4.0-h-tiny"
        logger.info(f"Loading tokenizer for model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info(f"Initialized Granite client with URL: {config.granite_url}")

    def _format_messages_with_template(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages using the tokenizer's chat template.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Formatted prompt string with proper role tags
        """
        # Apply the chat template using the tokenizer
        # This automatically adds the correct <|start_of_role|> tags
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            available_tools=self.tools if self.tools else None
        )
        return formatted_prompt

    def _convert_mcp_tools_to_granite_format(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tool definitions to Granite tool format.

        Combines hardcoded OpenStack tools (defined in OpenAPI format) with
        any additional MCP tools provided.

        Args:
            mcp_tools: List of MCP tool definitions

        Returns:
            List of Granite-formatted tool definitions
        """
        # Start with hardcoded OpenStack tools in OpenAPI format
        granite_tools = get_openstack_tools()
        logger.info(f"Loaded {len(granite_tools)} hardcoded OpenStack tools in OpenAPI format")

        # Add any additional MCP tools
        for mcp_tool in mcp_tools:
            try:
                # Convert to Granite tool format (similar to OpenAI function calling)
                granite_tool = {
                    "type": "function",
                    "function": {
                        "name": mcp_tool["name"],
                        "description": mcp_tool.get("description", ""),
                        "parameters": mcp_tool.get("inputSchema", {})
                    }
                }
                granite_tools.append(granite_tool)
                logger.debug(f"Converted MCP tool to Granite format: {mcp_tool['name']}")
            except Exception as e:
                logger.warning(f"Failed to convert MCP tool {mcp_tool.get('name', 'unknown')}: {e}")

        return granite_tools

    def _parse_text_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from text content.

        Granite models may return tool calls in various formats:
        1. XML tags: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        2. Plain JSON: {"name": "...", "arguments": {...}}
        3. Natural language: "I will call the X tool with {...}"

        Args:
            content: The text content from the assistant's response

        Returns:
            List of tool calls in OpenAI format, or None if no tool calls found
        """
        if not content or not self.tools:
            return None

        tool_calls = []

        # Build a set of valid tool names for quick lookup
        valid_tool_names = {t['function']['name'] for t in self.tools if 'function' in t}

        # Strategy 1: Extract from <tool_call> XML tags (most reliable)
        if "<tool_call>" in content:
            pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            for idx, match in enumerate(re.finditer(pattern, content, re.DOTALL)):
                json_str = match.group(1).strip()
                if json_str:
                    tool_call = self._parse_tool_call_json(json_str, idx, valid_tool_names)
                    if tool_call:
                        tool_calls.append(tool_call)
                        logger.info(f"Found tool call in XML tags: {tool_call['function']['name']}")

            # If we found XML tags, only use those (don't mix strategies)
            return tool_calls if tool_calls else None

        # Strategy 2: Look for standalone JSON objects with "name" field
        # Match JSON objects that contain a "name" field with a valid tool name
        for tool_name in valid_tool_names:
            # Pattern: {"name": "tool_name", "arguments": {...}} or {'name': 'tool_name', 'arguments': {...}}
            patterns = [
                rf'\{{\s*["\']name["\']\s*:\s*["\']({re.escape(tool_name)})["\']\s*,\s*["\']arguments["\']\s*:\s*(\{{[^}}]*\}})\s*\}}',
                rf'\{{\s*["\']arguments["\']\s*:\s*(\{{[^}}]*\}})\s*,\s*["\']name["\']\s*:\s*["\']({re.escape(tool_name)})["\']\s*\}}'
            ]

            for pattern in patterns:
                for match in re.finditer(pattern, content, re.DOTALL):
                    try:
                        # Extract the full JSON object
                        full_json = match.group(0)
                        tool_call = self._parse_tool_call_json(full_json, len(tool_calls), valid_tool_names)
                        if tool_call:
                            tool_calls.append(tool_call)
                            logger.info(f"Found tool call in JSON: {tool_call['function']['name']}")
                    except Exception as e:
                        logger.debug(f"Failed to parse JSON match: {e}")
                        continue

        if tool_calls:
            return tool_calls

        # Strategy 3: Natural language mentions with arguments
        # Pattern: "call the X tool" followed by JSON-like arguments
        for tool_name in valid_tool_names:
            pattern = rf'(?:call|execute|use|invoke)(?:\s+the)?\s+{re.escape(tool_name)}\s+tool[^{{]*(\{{[^}}]*\}})'
            for idx, match in enumerate(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)):
                args_str = match.group(1).strip()
                try:
                    arguments = json.loads(args_str)
                    if isinstance(arguments, dict):
                        tool_call = {
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments
                            }
                        }
                        tool_calls.append(tool_call)
                        logger.info(f"Found tool call in natural language: {tool_name}")
                except json.JSONDecodeError:
                    # Try Python literal eval as fallback
                    try:
                        arguments = ast.literal_eval(args_str)
                        if isinstance(arguments, dict):
                            tool_call = {
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": arguments
                                }
                            }
                            tool_calls.append(tool_call)
                            logger.info(f"Found tool call in natural language (literal_eval): {tool_name}")
                    except (ValueError, SyntaxError):
                        continue

        return tool_calls if tool_calls else None

    def _parse_tool_call_json(self, json_str: str, idx: int, valid_tool_names: set) -> Optional[Dict[str, Any]]:
        """Parse a single tool call JSON string.

        Args:
            json_str: JSON string to parse
            idx: Index for generating unique tool call ID
            valid_tool_names: Set of valid tool names to validate against

        Returns:
            Tool call in OpenAI format, or None if parsing failed
        """
        # Try parsing as JSON first, then Python dict syntax
        tool_call_json = None
        try:
            tool_call_json = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Granite sometimes uses Python dict syntax (single quotes)
                tool_call_json = ast.literal_eval(json_str)
            except (ValueError, SyntaxError) as e:
                logger.debug(f"Failed to parse tool call '{json_str[:100]}...': {e}")
                return None

        if not isinstance(tool_call_json, dict):
            logger.debug(f"Parsed value is not a dict: {type(tool_call_json)}")
            return None

        # Extract name and arguments, handling various formats
        tool_name = tool_call_json.get("name")
        arguments = tool_call_json.get("arguments", {})

        # Validate we have a tool name
        if not tool_name:
            logger.debug(f"Tool call missing 'name' field: {tool_call_json}")
            return None

        # Validate this is actually a known tool
        if tool_name not in valid_tool_names:
            logger.debug(f"Tool '{tool_name}' not in valid tool names")
            return None

        # Ensure arguments is a dict
        if not isinstance(arguments, dict):
            logger.debug(f"Arguments is not a dict for tool '{tool_name}': {type(arguments)}")
            arguments = {}

        # Convert to OpenAI format
        return {
            "id": f"call_{idx}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": arguments
            }
        }

    def start_chat(
        self,
        history: Optional[List] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        mcp_client: Optional[Any] = None,
    ) -> None:
        """Start a new chat session.

        Args:
            history: Optional conversation history to initialize the chat
            system_instruction: Optional system instruction to define agent behavior and constraints
            tools: Optional list of MCP tool definitions to make available to the LLM
            mcp_client: Optional MCP client instance for executing tool calls
        """
        # Initialize conversation with history if provided
        self.messages = history if history else []

        # Add system instruction as first message if provided
        # This aligns with Gemini's approach of setting it once at session start
        if system_instruction:
            logger.info("Starting chat session with system instruction")
            logger.debug(f"System instruction: {system_instruction}")
            self.messages.insert(0, {
                "role": "system",
                "content": system_instruction
            })

        # Store MCP client for tool execution
        self.mcp_client = mcp_client

        # Convert and store tools
        # Always include hardcoded OpenStack tools, plus any additional MCP tools
        if tools:
            logger.info(f"Starting chat session with {len(tools)} MCP tools")
            self.tools = self._convert_mcp_tools_to_granite_format(tools)
        else:
            # Even without MCP tools, load hardcoded OpenStack tools
            self.tools = get_openstack_tools()
            logger.info(f"Loaded {len(self.tools)} hardcoded OpenStack tools (no additional MCP tools)")

        logger.debug(f"Configured {len(self.tools)} total tools for Granite")

        logger.debug("Started new chat session")

    def send_message(self, message: str) -> str:
        """Send a message and get a response.

        Handles function calling loop if MCP tools are available.

        Args:
            message: The message to send

        Returns:
            The response text from Granite

        Raises:
            RuntimeError: If chat session is not started
        """
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": message
        })

        try:
            # Handle function calling loop
            max_iterations = 10  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                # Format messages using the chat template
                formatted_prompt = self._format_messages_with_template(self.messages)
                logger.debug(f"Formatted prompt (first 500 chars): {formatted_prompt[:500]}")

                # Build the request payload for completions endpoint
                payload = {
                    "prompt": formatted_prompt,
                    "temperature": self.config.granite_temperature,
                    "min_p": self.config.granite_min_p
                }

                # Add max_tokens if configured
                if self.config.granite_max_tokens:
                    payload["max_tokens"] = self.config.granite_max_tokens

                # Make API request to completions endpoint
                # Support full URLs (with /v1/completions) or base URLs
                url = self.config.granite_url
                if '/v1/chat/completions' in url:
                    url = url.replace('/v1/chat/completions', '/v1/completions')
                elif not url.endswith('/v1/completions'):
                    url = f"{url}/v1/completions"

                response = self.session.post(
                    url,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                result = response.json()

                # Store usage metadata if available
                if "usage" in result:
                    self.last_usage_metadata = result["usage"]
                    logger.debug(f"Usage metadata: {self.last_usage_metadata}")

                # Extract the assistant's response from completions endpoint
                if not result.get("choices") or len(result["choices"]) == 0:
                    raise RuntimeError("No response from Granite API")

                choice = result["choices"][0]
                # For completions endpoint, response is in 'text' field
                content = choice.get("text", "").strip()

                # Create assistant message for history
                assistant_message = {
                    "role": "assistant",
                    "content": content
                }

                # Add assistant message to history
                self.messages.append(assistant_message)

                # Check for text-based tool calls in the content
                if content:
                    logger.debug(f"Checking for text-based tool calls in content: {content[:200]}...")
                    tool_calls = self._parse_text_tool_calls(content)
                    if tool_calls:
                        logger.info(f"Found {len(tool_calls)} text-based tool calls")
                    else:
                        logger.debug("No text-based tool calls found")
                else:
                    tool_calls = None

                # If no tool calls, we're done - return the final response
                if not tool_calls:
                    # No tool calls, return the text response
                    # chat.py will display this final response
                    return content

                # We have tool calls to process
                # Prepare content for display (this is an intermediate message)
                # Remove tool call tags and clean up excessive whitespace
                display_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
                # Clean up multiple consecutive newlines (more than 2)
                display_content = re.sub(r'\n{3,}', '\n\n', display_content)
                # Strip leading/trailing whitespace
                display_content = display_content.strip()

                # Update the message in history to use clean content
                self.messages[-1]["content"] = display_content
                logger.debug(f"Cleaned content after tool call extraction: {display_content[:200]}...")

                # Display intermediate text content (before processing tool calls)
                # Only display if there's meaningful content after cleaning
                if display_content and len(display_content) > 10:
                    # Stop spinner before printing to avoid visual interference
                    from .spinner import get_global_spinner
                    spinner = get_global_spinner()
                    if spinner:
                        spinner.stop()

                    console.print("\n[green]Assistant:[/green]")
                    console.print(Panel(Markdown(display_content), border_style="blue"))
                    console.print()
                else:
                    # If there's no meaningful text content, just log that we're executing tools
                    logger.info(f"Executing {len(tool_calls)} tool call(s) without intermediate text")

                # Restart spinner for tool execution
                # The spinner will show while tools execute and while waiting for next LLM response
                from .spinner import get_global_spinner
                spinner = get_global_spinner()
                if spinner:
                    spinner.start()

                # Execute tool calls
                if not self.mcp_client:
                    logger.error("Tool calls requested but no MCP client available")
                    logger.error(f"Tool calls that were skipped: {[tc.get('function', {}).get('name') for tc in tool_calls]}")
                    return content

                # Only process the FIRST tool call to ensure sequential execution
                # The LLM will be called again after this tool completes, and can then
                # decide on the next step based on the result
                if len(tool_calls) > 1:
                    logger.warning(f"Found {len(tool_calls)} tool calls, but only executing first one for sequential execution")
                else:
                    logger.info(f"Processing 1 tool call")

                tool_call = tool_calls[0]
                logger.debug(f"Processing tool call: {tool_call}")
                function = tool_call.get("function", {})
                function_name = function.get("name")
                function_args = function.get("arguments", {})

                # Handle both string and dict arguments
                if isinstance(function_args, str):
                    try:
                        function_args = json.loads(function_args)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse function arguments: {e}")
                        function_args = {}
                elif not isinstance(function_args, dict):
                    logger.warning(f"Unexpected function arguments type: {type(function_args)}")
                    function_args = {}

                try:
                    logger.info(f"LLM wants to execute MCP tool: {function_name}")
                    logger.debug(f"Tool arguments: {function_args}")

                    # Execute the tool via MCP
                    def run_async_in_thread(coro):
                        """Run an async coroutine in a new thread with its own event loop."""
                        import threading
                        result_container = {}
                        exception_container = {}

                        def run_in_new_loop():
                            try:
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                try:
                                    result_container['result'] = new_loop.run_until_complete(coro)
                                finally:
                                    new_loop.close()
                            except Exception as e:
                                exception_container['exception'] = e

                        thread = threading.Thread(target=run_in_new_loop)
                        thread.start()
                        thread.join()

                        if 'exception' in exception_container:
                            raise exception_container['exception']
                        return result_container['result']

                    result = run_async_in_thread(
                        self.mcp_client.call_tool(function_name, function_args)
                    )

                    logger.debug(f"Tool result: {result}")

                    # Add tool result to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": function_name,
                        "content": str(result)
                    })

                except Exception as e:
                    logger.error(f"Error executing MCP tool {function_name}: {e}")
                    # Add error message to conversation
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": function_name,
                        "content": f"Error: {str(e)}"
                    })

                iteration += 1

            if iteration >= max_iterations:
                logger.warning("Maximum function calling iterations reached")
                return self.messages[-1].get("content", "")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Granite: {e}")
            raise RuntimeError(f"Granite API request failed: {e}")
        except Exception as e:
            logger.error(f"Error sending message to Granite: {e}")
            raise

    def send_message_stream(self, message: str) -> Iterator[str]:
        """Send a message and stream the response.

        Args:
            message: The message to send

        Yields:
            Chunks of the response text as they arrive

        Raises:
            RuntimeError: If streaming is not supported or request fails
        """
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": message
        })

        try:
            # Format messages using the chat template
            formatted_prompt = self._format_messages_with_template(self.messages)
            logger.debug(f"Formatted prompt for streaming (first 500 chars): {formatted_prompt[:500]}")

            # Build the request payload for completions endpoint
            payload = {
                "prompt": formatted_prompt,
                "stream": True,
                "temperature": self.config.granite_temperature,
                "min_p": self.config.granite_min_p
            }

            # Add max_tokens if configured
            if self.config.granite_max_tokens:
                payload["max_tokens"] = self.config.granite_max_tokens

            # Make streaming API request to completions endpoint
            # Support full URLs (with /v1/completions) or base URLs
            url = self.config.granite_url
            if '/v1/chat/completions' in url:
                url = url.replace('/v1/chat/completions', '/v1/completions')
            elif not url.endswith('/v1/completions'):
                url = f"{url}/v1/completions"

            response = self.session.post(
                url,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            full_content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break

                        try:
                            import json
                            chunk = json.loads(data)
                            if chunk.get("choices"):
                                choice = chunk["choices"][0]
                                # For completions endpoint, streaming uses "text" field
                                # For chat completions, it uses "delta" with "content"
                                content = choice.get("text", "")
                                if not content:
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                if content:
                                    full_content += content
                                    yield content
                        except json.JSONDecodeError:
                            continue

            # Add complete assistant message to history
            if full_content:
                self.messages.append({
                    "role": "assistant",
                    "content": full_content
                })

        except requests.exceptions.RequestException as e:
            logger.error(f"Error streaming message from Granite: {e}")
            raise RuntimeError(f"Granite streaming request failed: {e}")
        except Exception as e:
            logger.error(f"Error streaming message from Granite: {e}")
            raise

    def get_history(self) -> List:
        """Get the current chat history.

        Returns:
            List of messages in the conversation
        """
        return self.messages.copy()

    def clear_history(self) -> None:
        """Clear the chat history and start a new session."""
        self.messages = []
        logger.debug("Cleared chat history")

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Get usage metadata from the last response.

        Returns:
            Dictionary with token usage information, or None if not available
        """
        return self.last_usage_metadata if self.last_usage_metadata else None
