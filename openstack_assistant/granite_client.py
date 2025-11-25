"""Granite API client for the OpenStack Upgrade Assistant."""

import asyncio
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import Config

logger = logging.getLogger(__name__)
console = Console()


class GraniteClient:
    """Client for interacting with Granite LLM API.

    Attributes:
        config: Configuration object
        session: Requests session for API calls
        messages: Conversation history
        system_instruction: System instruction for the agent
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
        self.messages = []
        self.system_instruction = None
        self.tools = None
        self.mcp_client = None
        logger.info(f"Initialized Granite client with URL: {config.granite_url}")

    def _convert_mcp_tools_to_granite_format(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tool definitions to Granite tool format.

        Args:
            mcp_tools: List of MCP tool definitions

        Returns:
            List of Granite-formatted tool definitions
        """
        granite_tools = []
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
        """Parse XML-like tool call tags from text content.

        Some Granite models return tool calls as XML-like text tags instead of
        using structured function calling. This method extracts those calls.

        Args:
            content: The text content from the assistant's response

        Returns:
            List of tool calls in OpenAI format, or None if no tool calls found
        """
        if not content or "<tool_call>" not in content:
            return None

        tool_calls = []
        # Use regex to find all <tool_call>...</tool_call> blocks
        # Capture everything between tags (not just matching braces)
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.finditer(pattern, content, re.DOTALL)

        for idx, match in enumerate(matches):
            json_str = match.group(1).strip()

            if not json_str:
                continue

            # Check if JSON appears truncated (missing closing brace)
            is_truncated = False
            if json_str.count('{') > json_str.count('}'):
                logger.warning(f"Tool call appears truncated (unmatched braces): {json_str[:100]}...")
                is_truncated = True
                # Try to fix by adding missing closing braces
                missing_braces = json_str.count('{') - json_str.count('}')
                json_str += '}' * missing_braces
                logger.debug(f"Attempting to fix truncated JSON by adding {missing_braces} closing brace(s)")

            try:
                tool_call_json = json.loads(json_str)
                # Convert to OpenAI tool call format
                tool_call = {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_call_json.get("name"),
                        "arguments": tool_call_json.get("arguments", {})
                    }
                }
                tool_calls.append(tool_call)
                status = "truncated but recovered" if is_truncated else "complete"
                logger.info(f"Parsed text-based tool call ({status}): {tool_call['function']['name']} with args {tool_call['function']['arguments']}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call JSON '{json_str[:100]}...': {e}")
                # Log more context to help debug
                logger.debug(f"Full JSON string that failed to parse: {json_str}")
                continue

        return tool_calls if tool_calls else None

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

        # Store system instruction
        self.system_instruction = system_instruction
        if system_instruction:
            logger.info("Starting chat session with system instruction")
            logger.debug(f"System instruction: {system_instruction}")

        # Store MCP client for tool execution
        self.mcp_client = mcp_client

        # Convert and store tools if provided
        if tools:
            logger.info(f"Starting chat session with {len(tools)} MCP tools")
            self.tools = self._convert_mcp_tools_to_granite_format(tools)
            logger.debug(f"Configured {len(self.tools)} tools for Granite")

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
                # Build the request payload
                payload = {
                    "messages": self.messages.copy(),
                    "temperature": self.config.granite_temperature
                }

                # Add system instruction if set
                if self.system_instruction:
                    # Insert system message at the beginning
                    payload["messages"] = [
                        {"role": "system", "content": self.system_instruction}
                    ] + payload["messages"]

                # Add tools if available
                if self.tools:
                    payload["tools"] = self.tools

                # Make API request
                # Support full URLs (with /v1/chat/completions) or base URLs
                url = self.config.granite_url
                if not url.endswith('/v1/chat/completions'):
                    url = f"{url}/v1/chat/completions"

                response = self.session.post(
                    url,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                result = response.json()

                # Extract the assistant's response
                if not result.get("choices") or len(result["choices"]) == 0:
                    raise RuntimeError("No response from Granite API")

                choice = result["choices"][0]
                assistant_message = choice.get("message", {})

                # Add assistant message to history
                self.messages.append(assistant_message)

                # Check if there are tool calls (structured format)
                tool_calls = assistant_message.get("tool_calls")
                content = assistant_message.get("content", "")

                # If no structured tool calls, check for text-based tool calls
                if not tool_calls:
                    tool_calls = self._parse_text_tool_calls(content)

                    # If we found text-based tool calls, strip them from content
                    if tool_calls:
                        # Remove the <tool_call> tags from the content for cleaner display
                        clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                        # Update the message in history to use clean content
                        self.messages[-1]["content"] = clean_content
                        logger.debug(f"Cleaned content after tool call extraction: {clean_content}")
                        # Print the cleaned content before processing tool calls
                        if clean_content:
                            console.print("\n[green]Assistant:[/green]")
                            console.print(Panel(Markdown(clean_content), border_style="blue"))
                            console.print()
                elif content:
                    # Structured tool calls with text content - print the text first
                    console.print("\n[green]Assistant:[/green]")
                    console.print(Panel(Markdown(content), border_style="blue"))
                    console.print()

                if not tool_calls:
                    # No tool calls, return the text response
                    return assistant_message.get("content", "")

                # Execute tool calls
                if not self.mcp_client:
                    logger.error("Tool calls requested but no MCP client available")
                    return assistant_message.get("content", "")

                # Process each tool call
                for tool_call in tool_calls:
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
            # Build the request payload
            payload = {
                "messages": self.messages.copy(),
                "stream": True,
                "temperature": self.config.granite_temperature
            }

            # Add system instruction if set
            if self.system_instruction:
                payload["messages"] = [
                    {"role": "system", "content": self.system_instruction}
                ] + payload["messages"]

            # Add tools if available
            if self.tools:
                payload["tools"] = self.tools

            # Make streaming API request
            # Support full URLs (with /v1/chat/completions) or base URLs
            url = self.config.granite_url
            if not url.endswith('/v1/chat/completions'):
                url = f"{url}/v1/chat/completions"

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
                                delta = chunk["choices"][0].get("delta", {})
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
