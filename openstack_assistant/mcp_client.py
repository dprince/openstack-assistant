"""MCP (Model Context Protocol) client for the OpenStack Upgrade Assistant."""

import asyncio
import json
import logging
import subprocess
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for interacting with MCP servers.

    The Model Context Protocol (MCP) allows AI assistants to interact with
    external tools and data sources through a standardized interface.

    Attributes:
        server_command: Command to start the MCP server
        process: Subprocess running the MCP server
        tools: Available tools from the MCP server
        _notification_handler: Optional callback for handling server notifications
        _confirmation_handler: Optional callback for confirming tool execution
        _confirm_prefixes: List of tool name prefixes that require confirmation
        _allowed_tools: Optional list of allowed tool names (None = all tools allowed)
    """

    def __init__(self,
                 server_command: Optional[str] = None,
                 notification_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
                 confirmation_handler: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
                 confirm_prefixes: Optional[List[str]] = None,
                 allowed_tools: Optional[List[str]] = None):
        """Initialize the MCP client.

        Args:
            server_command: Command to start the MCP server (e.g., "npx @modelcontextprotocol/server-filesystem")
            notification_handler: Optional callback function for handling notifications from the server.
                                 The function receives a notification dict as its argument.
            confirmation_handler: Optional callback function for confirming tool execution.
                                 The function receives (tool_name, arguments) and returns bool.
            confirm_prefixes: List of tool name prefixes that require confirmation (e.g., ["create_", "watch_"])
            allowed_tools: Optional list of allowed tool names (None = all tools allowed)
        """
        self.server_command = server_command
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[Dict[str, Any]] = []
        self._message_id = 0
        self._notification_handler = notification_handler
        self._confirmation_handler = confirmation_handler
        self._confirm_prefixes = confirm_prefixes or []
        self._allowed_tools = allowed_tools
        logger.info(f"Initialized MCP client with command: {server_command}")
        if self._confirm_prefixes:
            logger.info(f"Tool confirmation required for prefixes: {self._confirm_prefixes}")
        if self._allowed_tools:
            logger.info(f"Allowed tools filter: {self._allowed_tools}")

    async def connect(self) -> None:
        """Connect to the MCP server.

        Starts the MCP server process if a command is provided.

        Raises:
            RuntimeError: If connection fails
        """
        if not self.server_command:
            logger.warning("No MCP server command provided, skipping connection")
            return

        try:
            # Start the MCP server process
            self.process = subprocess.Popen(
                self.server_command.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            logger.info("Started MCP server process")

            # Initialize the connection
            await self._initialize()

            # List available tools
            await self._list_tools()

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise RuntimeError(f"MCP server connection failed: {e}")

    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {},
                "clientInfo": {
                    "name": "openstack-upgrade-assistant",
                    "version": "0.1.0",
                },
            },
        }
        response = await self._send_request(request)
        logger.debug(f"Initialized MCP connection: {response}")

    async def _list_tools(self) -> None:
        """List available tools from the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/list",
            "params": {},
        }
        response = await self._send_request(request)

        if "result" in response and "tools" in response["result"]:
            all_tools = response["result"]["tools"]

            # Filter tools if allowed_tools list is specified
            if self._allowed_tools:
                self.tools = [tool for tool in all_tools if tool.get('name') in self._allowed_tools]
                logger.info(f"Found {len(all_tools)} tools from MCP server, filtered to {len(self.tools)} allowed tools")
                filtered_out = len(all_tools) - len(self.tools)
                if filtered_out > 0:
                    logger.info(f"Filtered out {filtered_out} tools not in allowed list")
            else:
                self.tools = all_tools
                logger.info(f"Found {len(self.tools)} tools from MCP server")

            for tool in self.tools:
                logger.debug(f"  - {tool.get('name')}: {tool.get('description')}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            The result from the tool execution

        Raises:
            ValueError: If the tool is not found or user denies confirmation
            RuntimeError: If the tool call fails
        """
        if not any(tool["name"] == tool_name for tool in self.tools):
            raise ValueError(f"Tool '{tool_name}' not found in available tools")

        # Check if tool requires confirmation
        requires_confirmation = any(tool_name.startswith(prefix) for prefix in self._confirm_prefixes)
        if requires_confirmation and self._confirmation_handler:
            logger.info(f"Tool '{tool_name}' requires user confirmation")
            if not self._confirmation_handler(tool_name, arguments):
                logger.info(f"User denied execution of tool '{tool_name}'")
                raise ValueError(f"User denied execution of tool '{tool_name}'")

        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        try:
            response = await self._send_request(request)
            if "result" in response:
                return response["result"]
            elif "error" in response:
                raise RuntimeError(f"Tool call error: {response['error']}")
            return response
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP server and handle notifications.

        This method sends a request and then loops reading messages from the server.
        Notifications (messages without an 'id' field) are handled via _handle_notification,
        and the loop continues until the response with matching ID is received.

        Args:
            request: The JSON-RPC request

        Returns:
            The JSON-RPC response

        Raises:
            RuntimeError: If the server is not running or communication fails
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("MCP server is not running")

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            logger.debug(f"Sent request: {request}")

            # Loop reading messages until we get the response
            request_id = request.get("id")
            while True:
                response_line = self.process.stdout.readline()
                if not response_line:
                    raise RuntimeError("No response from MCP server")

                message = json.loads(response_line)
                logger.debug(f"Received message: {message}")

                # Check if it's a notification (no "id" field) or response (has "id")
                if "id" not in message:
                    # It's a notification - handle it and continue waiting
                    self._handle_notification(message)
                elif message.get("id") == request_id:
                    # It's the response we're waiting for
                    return message
                else:
                    # It's a response for a different request (shouldn't happen in our case)
                    logger.warning(f"Received unexpected response with id {message.get('id')}, expected {request_id}")

        except Exception as e:
            logger.error(f"Error communicating with MCP server: {e}")
            raise RuntimeError(f"MCP communication error: {e}")

    def _get_next_id(self) -> int:
        """Get the next message ID.

        Returns:
            The next message ID
        """
        self._message_id += 1
        return self._message_id

    def _handle_notification(self, notification: Dict[str, Any]) -> None:
        """Handle a notification from the MCP server.

        Args:
            notification: The notification message from the server
        """
        method = notification.get("method", "")
        params = notification.get("params", {})

        # If user provided a custom notification handler, use it
        if self._notification_handler:
            self._notification_handler(notification)
        else:
            # Default behavior: print to console for CLI visibility
            if method == "notifications/message":
                level = params.get("level", "info")
                message = params.get("message", "")
                # Print to console so it appears in the CLI
                print(f"[{level.upper()}] {message}")
            else:
                logger.info(f"Notification {method}: {params}")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools.

        Returns:
            List of tool definitions
        """
        return self.tools

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Disconnected from MCP server")

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if self.process:
            self.process.terminate()
