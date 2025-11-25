"""Interactive chat interface for the OpenStack Upgrade Assistant."""

import asyncio
import logging
import sys
from typing import Optional, Union

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import Config
from .gemini_client import GeminiClient
from .granite_client import GraniteClient
from .mcp_client import MCPClient
from .spinner import StatusSpinner, set_global_spinner

logger = logging.getLogger(__name__)


class ChatInterface:
    """Interactive chat interface.

    Provides a REPL-style chat interface for interacting with the assistant.

    Attributes:
        llm_client: LLM client (Gemini or Granite) for AI interactions
        mcp_client: Optional MCP client for tool access
        config: Configuration object
        console: Rich console for formatted output
        session: Prompt toolkit session for input
    """

    def __init__(
        self,
        llm_client: Union[GeminiClient, GraniteClient],
        config: Config,
        mcp_client: Optional[MCPClient] = None,
    ):
        """Initialize the chat interface.

        Args:
            llm_client: LLM client (Gemini or Granite) for AI interactions
            config: Configuration object
            mcp_client: Optional MCP client for tool access
        """
        self.llm_client = llm_client
        self.config = config
        self.mcp_client = mcp_client
        self.console = Console()
        self.session: PromptSession = PromptSession(history=InMemoryHistory())

    async def start(self) -> None:
        """Start the interactive chat session.

        Displays welcome message and enters the chat loop.
        """
        self._display_welcome()

        # Load system instruction if provided
        system_instruction = self._load_system_instruction()

        # Get MCP tools if connected
        mcp_tools = None
        if self.mcp_client:
            mcp_tools = self.mcp_client.get_available_tools()

        # Start LLM chat session with system instruction and MCP tools
        self.llm_client.start_chat(
            system_instruction=system_instruction,
            tools=mcp_tools,
            mcp_client=self.mcp_client
        )

        # Display agent identity if configured
        if system_instruction:
            self.console.print("[cyan]Agent identity configured from system instruction file[/cyan]\n")

        # Display available MCP tools if connected
        #if self.mcp_client:
            #self._display_mcp_tools()
            #if mcp_tools:
                #self.console.print("[cyan]MCP tools are now available to the LLM for autonomous use[/cyan]\n")

        # Display initial AI greeting
        self._display_initial_greeting()

        # Enter chat loop
        try:
            await self._chat_loop()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Chat session interrupted.[/yellow]")
        except EOFError:
            self.console.print("\n[yellow]Exiting chat.[/yellow]")
        finally:
            self._display_goodbye()

    def _display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = """
# OpenStack Assistant

Welcome to the interactive chat interface!

**Available commands:**
- `.exit` or `.quit` - Exit the chat
- `.clear` - Clear conversation history
- `.tools` - Show available MCP tools (if connected)
- `.help` - Show this help message

Type your questions and press Enter to chat with the assistant.
        """
        self.console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="green"))

    def _display_goodbye(self) -> None:
        """Display goodbye message."""
        self.console.print("\n[green]Thank you for using OpenStack Upgrade Assistant![/green]")

    def _display_initial_greeting(self) -> None:
        """Display initial AI greeting.

        Triggers the AI to auto-execute step 1 as specified in the system instruction.
        """
        logger.info("Displaying initial AI greeting")
        try:
            # Use spinner for initial message too
            spinner = StatusSpinner("Thinking")
            set_global_spinner(spinner)

            with spinner:
                response = self.llm_client.send_message("Tell me the state of my openstackversion resrouce. Then lets continue the upgrade process?")

            # Clear the global spinner reference
            set_global_spinner(None)

            self.console.print("[green]Assistant:[/green]")
            self.console.print(Panel(Markdown(response), border_style="blue"))
            self.console.print()

        except Exception as e:
            logger.error(f"Error getting initial response: {e}")
            # Make sure spinner is stopped on error
            from .spinner import stop_global_spinner
            stop_global_spinner()
            # Don't fail the whole chat if initial response fails, just log and continue
            logger.warning("Continuing chat session without initial response")

    def _display_mcp_tools(self) -> None:
        """Display available MCP tools."""
        if not self.mcp_client:
            return

        tools = self.mcp_client.get_available_tools()
        if not tools:
            return

        self.console.print("\n[cyan]Available MCP Tools:[/cyan]")
        for tool in tools:
            name = tool.get("name", "Unknown")
            description = tool.get("description", "No description")
            self.console.print(f"  • [bold]{name}[/bold]: {description}")
        self.console.print()

    def _load_system_instruction(self) -> Optional[str]:
        """Load system instruction from file if configured.

        Returns:
            System instruction text if file is configured and exists, None otherwise
        """
        if not self.config.system_instruction_file:
            return None

        try:
            instruction_path = self.config.system_instruction_file
            if not instruction_path.exists():
                logger.warning(f"System instruction file not found: {instruction_path}")
                return None

            with open(instruction_path, "r") as f:
                instruction = f.read().strip()

            if not instruction:
                logger.warning(f"System instruction file is empty: {instruction_path}")
                return None

            # Inject namespace defaults if namespace is configured
            if self.config.namespace:
                namespace_defaults = f"## Defaults\nDefault to use the '{self.config.namespace}' k8s namespace unless otherwise instructed when using any MCP tools."
                instruction = f"{namespace_defaults}\n\n{instruction}"
                logger.info(f"Injected namespace defaults for: {self.config.namespace}")

            logger.info(f"Loaded system instruction from: {instruction_path}")
            return instruction

        except Exception as e:
            logger.error(f"Failed to load system instruction: {e}")
            return None

    async def _chat_loop(self) -> None:
        """Main chat loop.

        Continuously prompts for user input and displays AI responses.
        """
        while True:
            try:
                # Get user input using async prompt
                with patch_stdout():
                    user_input = await self.session.prompt_async(">>> ")
                    user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in [".exit", ".quit"]:
                    break
                elif user_input.lower() == ".clear":
                    self._clear_history()
                    continue
                elif user_input.lower() == ".tools":
                    self._display_mcp_tools()
                    continue
                elif user_input.lower() == ".help":
                    self._display_welcome()
                    continue

                # Send message to Gemini and display response
                self._handle_message(user_input)

            except KeyboardInterrupt:
                # Ctrl+C - ask for confirmation
                try:
                    with patch_stdout():
                        confirm = await self.session.prompt_async("\nAre you sure you want to exit? (y/n): ")
                    if confirm.lower() in ["y", "yes"]:
                        break
                except (KeyboardInterrupt, EOFError):
                    break
            except EOFError:
                # Ctrl+D - exit
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                self.console.print(f"[red]Error: {e}[/red]")

    def _handle_message(self, message: str) -> None:
        """Handle a user message.

        Args:
            message: The user's message
        """
        try:
            # Use spinner to show status while waiting for response
            # The spinner will automatically stop if MCP confirmation is needed
            spinner = StatusSpinner("Thinking")
            set_global_spinner(spinner)

            with spinner:
                response = self.llm_client.send_message(message)

            # Clear the global spinner reference
            set_global_spinner(None)

            # Display response
            self.console.print("\n[green]Assistant:[/green]")
            self.console.print(Panel(Markdown(response), border_style="blue"))
            self.console.print()

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            # Make sure spinner is stopped on error
            from .spinner import get_global_spinner, stop_global_spinner
            stop_global_spinner()
            self.console.print(f"[red]Error: Failed to get response from LLM. {e}[/red]")

    def _clear_history(self) -> None:
        """Clear the conversation history."""
        self.llm_client.clear_history()
        self.console.print("[yellow]Conversation history cleared.[/yellow]")
