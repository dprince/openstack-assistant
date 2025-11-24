"""Gemini API client for the OpenStack Upgrade Assistant."""

import asyncio
import logging
import sys
from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)


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
            response = self.chat_session.send_message(message)

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

                # Look for function call in the parts
                function_call = None
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        break

                if not function_call:
                    # No function call, return the text response
                    break

                # Execute the function call via MCP
                if not self.mcp_client:
                    logger.error("Function call requested but no MCP client available")
                    break

                # Ask user for confirmation before executing the tool
                logger.info(f"LLM wants to execute MCP tool: {function_call.name}")
                logger.debug(f"Tool arguments: {dict(function_call.args)}")

                # Format the tool call for user review
                print(f"\n[Tool Call Request]")
                print(f"Tool: {function_call.name}")
                print(f"Arguments: {dict(function_call.args)}")

                # Prompt user for confirmation
                user_approved = False
                while True:
                    try:
                        user_response = input("\nProceed with this tool call? [y/n]: ").strip().lower()
                        if user_response in ['y', 'yes']:
                            user_approved = True
                            break
                        elif user_response in ['n', 'no']:
                            # User rejected the tool call - send error back to model
                            logger.info("User rejected tool call")
                            error_response = types.Part(
                                function_response=types.FunctionResponse(
                                    name=function_call.name,
                                    response={"error": "User rejected tool execution"}
                                )
                            )
                            response = self.chat_session.send_message(error_response)
                            iteration += 1
                            # Break out of confirmation loop and continue with outer while loop
                            break
                        else:
                            print("Please enter 'y' or 'n'")
                            continue
                    except (EOFError, KeyboardInterrupt):
                        logger.info("User interrupted tool call confirmation")
                        print("\nTool call cancelled")
                        error_response = types.Part(
                            function_response=types.FunctionResponse(
                                name=function_call.name,
                                response={"error": "User cancelled tool execution"}
                            )
                        )
                        response = self.chat_session.send_message(error_response)
                        iteration += 1
                        break

                # If user rejected, continue to next iteration
                if not user_approved:
                    continue

                try:
                    logger.info(f"Executing MCP tool: {function_call.name}")

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
                    iteration += 1

            if iteration >= max_iterations:
                logger.warning("Maximum function calling iterations reached")

            return response.text

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
