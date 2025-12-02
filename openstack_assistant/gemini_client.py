"""Gemini API client for the OpenStack Upgrade Assistant."""

import ast
import asyncio
import json
import logging
import re
import sys
from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import Config
from .message_logger import MessageLogger

logger = logging.getLogger(__name__)
console = Console()


class GeminiClient:
    """Client for interacting with Google's Gemini API.

    Attributes:
        config: Configuration object
        client: Gemini client instance
        chat_session: Active chat session for conversation history
    """

    def __init__(self, config: Config):
        """Initialize the Gemini client.

        Args:
            config: Configuration object containing API key and model name
        """
        self.config = config
        self.client = genai.Client(
            api_key=config.gemini_api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.chat_session = None
        self.mcp_client = None  # Will be set when tools are provided
        self.last_usage_metadata = None  # Store usage metadata from last response
        self.message_logger = MessageLogger(config.raw_message_log_dir)
        logger.info(f"Initialized Gemini client with model: {config.gemini_model}")

    def _convert_mcp_tools_to_gemini_format(self, mcp_tools: List[Dict[str, Any]]) -> List[types.Tool]:
        """Convert MCP tool definitions to Gemini Tool format.

        Args:
            mcp_tools: List of MCP tool definitions

        Returns:
            List of Gemini Tool objects
        """
        gemini_tools = []
        for mcp_tool in mcp_tools:
            try:
                # Create function declaration from MCP tool
                function_declaration = types.FunctionDeclaration(
                    name=mcp_tool["name"],
                    description=mcp_tool.get("description", ""),
                    parameters=mcp_tool.get("inputSchema", {})
                )
                gemini_tools.append(types.Tool(function_declarations=[function_declaration]))
                logger.debug(f"Converted MCP tool to Gemini format: {mcp_tool['name']}")
            except Exception as e:
                logger.warning(f"Failed to convert MCP tool {mcp_tool.get('name', 'unknown')}: {e}")

        return gemini_tools

    def _parse_text_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse XML-like tool call tags from text content.

        Some models return tool calls as XML-like text tags instead of
        using structured function calling. This method extracts those calls.

        Args:
            content: The text content from the assistant's response

        Returns:
            List of tool call dictionaries, or None if no tool calls found
        """
        if not content or "<tool_call>" not in content:
            return None

        tool_calls = []
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            json_str = match.group(1).strip()
            if not json_str:
                continue

            # Try parsing as JSON first, then Python dict syntax
            tool_call_json = None
            try:
                tool_call_json = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    # Fallback to Python dict syntax (single quotes)
                    tool_call_json = ast.literal_eval(json_str)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Failed to parse tool call '{json_str[:100]}...': {e}")
                    continue

            # Validate required fields
            tool_name = tool_call_json.get("name") if isinstance(tool_call_json, dict) else None
            if not tool_name:
                logger.error(f"Tool call missing 'name' field: {tool_call_json}")
                continue

            tool_calls.append(tool_call_json)
            logger.info(f"Parsed tool call: {tool_name}")

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
            history: Optional conversation history to initialize the chat (not supported in new SDK)
            system_instruction: Optional system instruction to define agent behavior and constraints
            tools: Optional list of MCP tool definitions to make available to the LLM
            mcp_client: Optional MCP client instance for executing tool calls
        """
        if history:
            logger.warning("Chat history initialization is not supported in the new SDK")

        # Store MCP client for tool execution
        self.mcp_client = mcp_client

        # Ensure model name has the 'models/' prefix for v1beta API
        model_name = self.config.gemini_model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        # Create config with system instruction and/or tools if provided
        config_params = {"model": model_name}
        if system_instruction or tools:
            generate_config = types.GenerateContentConfig()

            if system_instruction:
                logger.info("Starting chat session with system instruction")
                logger.debug(f"System instruction: {system_instruction}")
                generate_config.system_instruction = system_instruction

            if tools:
                logger.info(f"Starting chat session with {len(tools)} MCP tools")
                gemini_tools = self._convert_mcp_tools_to_gemini_format(tools)
                if gemini_tools:
                    generate_config.tools = gemini_tools
                    logger.debug(f"Configured {len(gemini_tools)} tools for Gemini")

            config_params["config"] = generate_config

        self.chat_session = self.client.chats.create(**config_params)
        logger.debug("Started new chat session")

    def send_message(self, message: str) -> str:
        """Send a message and get a response.

        Handles function calling loop if MCP tools are available.

        Args:
            message: The message to send

        Returns:
            The response text from Gemini

        Raises:
            RuntimeError: If chat session is not started
        """
        if not self.chat_session:
            self.start_chat()

        try:
            # Log the request
            self.message_logger.log_request(
                {"message": message},
                metadata={"model": self.config.gemini_model, "client": "gemini"}
            )

            response = self.chat_session.send_message(message)

            # Accumulate usage across all iterations for this conversation turn
            accumulated_usage = {
                'prompt_token_count': 0,
                'candidates_token_count': 0,
                'total_token_count': 0
            }

            # Add usage from initial response
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                accumulated_usage['prompt_token_count'] += getattr(usage, 'prompt_token_count', 0)
                accumulated_usage['candidates_token_count'] += getattr(usage, 'candidates_token_count', 0)
                accumulated_usage['total_token_count'] += getattr(usage, 'total_token_count', 0)
                logger.debug(f"Initial response usage: prompt={accumulated_usage['prompt_token_count']}, completion={accumulated_usage['candidates_token_count']}, total={accumulated_usage['total_token_count']}")

            # Handle function calling loop
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            while iteration < max_iterations:
                # Check if response contains function calls
                if not response.candidates:
                    break

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    break

                # Look for function call and text in the parts
                function_call = None
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                    elif hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                # Combine all text parts
                text_content = ''.join(text_parts).strip()

                # Check for text-based tool calls in the content (if no structured function call)
                text_tool_calls = None
                if not function_call and text_content:
                    text_tool_calls = self._parse_text_tool_calls(text_content)

                # Check if we have any tool calls to process
                has_tool_calls = function_call or text_tool_calls
                if not has_tool_calls:
                    # No tool calls, we're done with this iteration
                    # Don't display here - let chat.py display the final response
                    break

                # We have tool calls to process
                # Prepare intermediate content for display
                display_content = text_content
                if text_tool_calls:
                    # Remove the XML tags for cleaner display
                    display_content = re.sub(r'<tool_call>.*?</tool_call>', '', text_content, flags=re.DOTALL).strip()
                    logger.debug(f"Cleaned content after tool call extraction: {display_content}")

                # Display intermediate text content (before processing tool calls)
                if display_content:
                    # Stop spinner before printing to avoid visual interference
                    from .spinner import get_global_spinner
                    spinner = get_global_spinner()
                    if spinner:
                        spinner.stop()

                    console.print("\n[green]Assistant:[/green]")
                    console.print(Panel(Markdown(display_content), border_style="blue"))
                    console.print()

                # Restart spinner for tool execution
                # The spinner will show while tools execute and while waiting for next LLM response
                from .spinner import get_global_spinner
                spinner = get_global_spinner()
                if spinner:
                    spinner.start()

                # Execute tool calls via MCP
                if not self.mcp_client:
                    logger.error("Tool call requested but no MCP client available")
                    break

                # Process structured function call
                if function_call:
                    try:
                        logger.info(f"LLM wants to execute MCP tool: {function_call.name}")
                        logger.debug(f"Tool arguments: {dict(function_call.args)}")

                        # Execute the tool via MCP
                        # We need to handle the case where we're already in an event loop
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
                            self.mcp_client.call_tool(
                                function_call.name,
                                dict(function_call.args)
                            )
                        )

                        logger.debug(f"Tool result: {result}")

                        # Send the function result back to Gemini
                        function_response = types.Part(
                            function_response=types.FunctionResponse(
                                name=function_call.name,
                                response={"result": result}
                            )
                        )

                        response = self.chat_session.send_message(function_response)

                        # Accumulate usage from tool response
                        if hasattr(response, 'usage_metadata') and response.usage_metadata:
                            usage = response.usage_metadata
                            accumulated_usage['prompt_token_count'] += getattr(usage, 'prompt_token_count', 0)
                            accumulated_usage['candidates_token_count'] += getattr(usage, 'candidates_token_count', 0)
                            accumulated_usage['total_token_count'] += getattr(usage, 'total_token_count', 0)
                            logger.debug(f"Iteration {iteration} usage: prompt={getattr(usage, 'prompt_token_count', 0)}, completion={getattr(usage, 'candidates_token_count', 0)}")
                            logger.debug(f"Accumulated so far: prompt={accumulated_usage['prompt_token_count']}, completion={accumulated_usage['candidates_token_count']}, total={accumulated_usage['total_token_count']}")

                        iteration += 1

                    except Exception as e:
                        logger.error(f"Error executing MCP tool {function_call.name}: {e}")
                        # Send error back to the model
                        error_response = types.Part(
                            function_response=types.FunctionResponse(
                                name=function_call.name,
                                response={"error": str(e)}
                            )
                        )
                        response = self.chat_session.send_message(error_response)

                        # Accumulate usage from error response
                        if hasattr(response, 'usage_metadata') and response.usage_metadata:
                            usage = response.usage_metadata
                            accumulated_usage['prompt_token_count'] += getattr(usage, 'prompt_token_count', 0)
                            accumulated_usage['candidates_token_count'] += getattr(usage, 'candidates_token_count', 0)
                            accumulated_usage['total_token_count'] += getattr(usage, 'total_token_count', 0)
                            logger.debug(f"Error iteration {iteration} usage accumulated")

                        iteration += 1

                # Process text-based tool calls
                elif text_tool_calls:
                    for tool_call in text_tool_calls:
                        try:
                            tool_name = tool_call.get("name")
                            tool_args = tool_call.get("arguments", {})

                            logger.info(f"LLM wants to execute MCP tool (text-based): {tool_name}")
                            logger.debug(f"Tool arguments: {tool_args}")

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
                                self.mcp_client.call_tool(tool_name, tool_args)
                            )

                            logger.debug(f"Tool result: {result}")

                            # Send the function result back to Gemini
                            function_response = types.Part(
                                function_response=types.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result}
                                )
                            )

                            response = self.chat_session.send_message(function_response)

                            # Accumulate usage from text-based tool response
                            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                                usage = response.usage_metadata
                                accumulated_usage['prompt_token_count'] += getattr(usage, 'prompt_token_count', 0)
                                accumulated_usage['candidates_token_count'] += getattr(usage, 'candidates_token_count', 0)
                                accumulated_usage['total_token_count'] += getattr(usage, 'total_token_count', 0)

                        except Exception as e:
                            logger.error(f"Error executing MCP tool {tool_name}: {e}")
                            # Send error back to the model
                            error_response = types.Part(
                                function_response=types.FunctionResponse(
                                    name=tool_name,
                                    response={"error": str(e)}
                                )
                            )
                            response = self.chat_session.send_message(error_response)

                            # Accumulate usage from error response
                            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                                usage = response.usage_metadata
                                accumulated_usage['prompt_token_count'] += getattr(usage, 'prompt_token_count', 0)
                                accumulated_usage['candidates_token_count'] += getattr(usage, 'candidates_token_count', 0)
                                accumulated_usage['total_token_count'] += getattr(usage, 'total_token_count', 0)

                    iteration += 1

            if iteration >= max_iterations:
                logger.warning("Maximum function calling iterations reached")

            # Store accumulated usage metadata
            # Convert the accumulated dict to a format similar to the SDK's usage_metadata object
            class AccumulatedUsage:
                def __init__(self, prompt_tokens, completion_tokens, total_tokens):
                    self.prompt_token_count = prompt_tokens
                    self.candidates_token_count = completion_tokens
                    self.total_token_count = total_tokens

            self.last_usage_metadata = AccumulatedUsage(
                accumulated_usage['prompt_token_count'],
                accumulated_usage['candidates_token_count'],
                accumulated_usage['total_token_count']
            )
            logger.info(f"Final accumulated usage: prompt={accumulated_usage['prompt_token_count']}, completion={accumulated_usage['candidates_token_count']}, total={accumulated_usage['total_token_count']}")

            # Extract text from response, even if it contains function calls
            # This handles the case where we hit max iterations with pending function calls
            response_text = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                response_text = ''.join(text_parts).strip()
            else:
                # Fallback to response.text (may trigger warning if function calls present)
                response_text = response.text

            # Log the final response
            response_data = {
                "text": response_text,
                "candidates": []
            }
            # Try to capture the full response structure for debugging
            if response.candidates:
                for candidate in response.candidates:
                    candidate_data = {}
                    if candidate.content and candidate.content.parts:
                        candidate_data["parts"] = []
                        for part in candidate.content.parts:
                            part_data = {}
                            if hasattr(part, 'text') and part.text:
                                part_data["text"] = part.text
                            if hasattr(part, 'function_call') and part.function_call:
                                part_data["function_call"] = {
                                    "name": part.function_call.name,
                                    "args": dict(part.function_call.args) if part.function_call.args else {}
                                }
                            candidate_data["parts"].append(part_data)
                    response_data["candidates"].append(candidate_data)

            self.message_logger.log_response(
                response_data,
                metadata={
                    "model": self.config.gemini_model,
                    "client": "gemini",
                    "usage": accumulated_usage
                }
            )

            return response_text

        except Exception as e:
            logger.error(f"Error sending message to Gemini: {e}")
            raise

    def send_message_stream(self, message: str) -> Iterator[str]:
        """Send a message and stream the response.

        Args:
            message: The message to send

        Yields:
            Chunks of the response text as they arrive

        Raises:
            RuntimeError: If chat session is not started
        """
        if not self.chat_session:
            self.start_chat()

        try:
            for chunk in self.chat_session.send_message_stream(message):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error streaming message from Gemini: {e}")
            raise

    def get_history(self) -> List:
        """Get the current chat history.

        Returns:
            List of messages in the conversation (not supported in new SDK)
        """
        logger.warning("Chat history retrieval is not fully supported in the new SDK")
        if not self.chat_session:
            return []
        # The new SDK doesn't expose history directly, return empty list
        return []

    def clear_history(self) -> None:
        """Clear the chat history and start a new session."""
        self.start_chat()
        logger.debug("Cleared chat history")

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Get usage metadata from the last response.

        Returns:
            Dictionary with token usage information, or None if not available
        """
        if not self.last_usage_metadata:
            return None

        usage_dict = {}
        if hasattr(self.last_usage_metadata, 'prompt_token_count'):
            usage_dict['prompt_tokens'] = self.last_usage_metadata.prompt_token_count
        if hasattr(self.last_usage_metadata, 'candidates_token_count'):
            usage_dict['completion_tokens'] = self.last_usage_metadata.candidates_token_count
        if hasattr(self.last_usage_metadata, 'total_token_count'):
            usage_dict['total_tokens'] = self.last_usage_metadata.total_token_count

        return usage_dict if usage_dict else None
