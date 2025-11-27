"""Command-line interface for the OpenStack Upgrade Assistant."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from . import __version__
from .chat import ChatInterface
from .config import Config
from .gemini_client import GeminiClient
from .granite_client import GraniteClient
from .mcp_client import MCPClient

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


def create_llm_client(config: Config, provider: Optional[str] = None):
    """Create the appropriate LLM client based on configuration.

    Args:
        config: Configuration object
        provider: Optional provider name ('gemini' or 'granite'). If not specified,
                  auto-detects based on available credentials.

    Returns:
        Either GeminiClient or GraniteClient instance

    Raises:
        ValueError: If no LLM provider is configured or requested provider is not configured
    """
    # If provider is explicitly specified, use it
    if provider == "gemini":
        if not config.gemini_api_key:
            raise ValueError(
                "Gemini provider requested but GEMINI_API_KEY is not configured"
            )
        logger.info("Using Gemini LLM client")
        return GeminiClient(config)
    elif provider == "granite":
        if not (config.granite_url and config.granite_user_key):
            raise ValueError(
                "Granite provider requested but GRANITE_URL and GRANITE_USER_KEY are not configured"
            )
        logger.info("Using Granite LLM client")
        return GraniteClient(config)

    # Auto-detect: Prefer Granite if configured, otherwise use Gemini
    if config.granite_url and config.granite_user_key:
        logger.info("Using Granite LLM client")
        return GraniteClient(config)
    elif config.gemini_api_key:
        logger.info("Using Gemini LLM client")
        return GeminiClient(config)
    else:
        raise ValueError(
            "No LLM provider configured. "
            "Please set either GRANITE_URL/GRANITE_USER_KEY or GEMINI_API_KEY"
        )


def setup_parser() -> argparse.ArgumentParser:
    """Setup command-line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="OpenStack Upgrade Assistant - AI-powered chat interface for OpenStack upgrades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file for debug output (enables automatic DEBUG level logging to file)",
    )

    # MCP arguments
    parser.add_argument(
        "--mcp-server",
        type=str,
        help="Command to start MCP server (e.g., 'npx @modelcontextprotocol/server-filesystem /path')",
    )

    # LLM Configuration
    llm_group = parser.add_argument_group("LLM provider options")

    llm_group.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "granite"],
        help="LLM provider to use (auto-detected from environment if not specified)",
    )

    llm_group.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (overrides GEMINI_API_KEY environment variable)",
    )

    llm_group.add_argument(
        "--granite-url",
        type=str,
        help="Granite API URL (overrides GRANITE_URL environment variable)",
    )

    llm_group.add_argument(
        "--granite-key",
        type=str,
        help="Granite user key (overrides GRANITE_USER_KEY environment variable)",
    )

    llm_group.add_argument(
        "--model",
        type=str,
        help="Model to use (overrides GEMINI_MODEL or GRANITE_MODEL)",
    )

    # Namespace argument
    parser.add_argument(
        "--namespace",
        type=str,
        help="Default Kubernetes namespace to use (overrides NAMESPACE environment variable)",
    )

    return parser


async def run_chat_mode(
    config: Config,
    mcp_command: Optional[str] = None,
    provider: Optional[str] = None,
) -> int:
    """Run interactive chat mode.

    Args:
        config: Configuration object
        mcp_command: Optional MCP server command
        provider: Optional LLM provider ('gemini' or 'granite')

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Initialize LLM client
        llm_client = create_llm_client(config, provider)

        # Initialize MCP client if provided
        mcp_client = None
        if mcp_command:
            console = Console()
            console.print("[cyan]Connecting to MCP server...[/cyan]")

            # Create notification handler that uses Rich console
            def notification_handler(notification):
                method = notification.get("method", "")
                params = notification.get("params", {})
                if method == "notifications/message":
                    level = params.get("level", "info")
                    message = params.get("message", "")
                    # Use Rich console for formatted output
                    level_colors = {
                        "debug": "dim",
                        "info": "cyan",
                        "warning": "yellow",
                        "error": "red"
                    }
                    color = level_colors.get(level.lower(), "white")
                    console.print(f"[{color}][MCP {level.upper()}] {message}[/{color}]")

            # Create confirmation handler for chat mode
            def confirmation_handler(tool_name: str, arguments: dict) -> bool:
                # Import here to avoid circular dependency
                from .spinner import get_global_spinner

                # Stop the spinner before showing confirmation prompt
                # This ensures the prompt appears on a clean new line
                spinner = get_global_spinner()
                if spinner:
                    spinner.stop()

                console.print(f"\n[yellow]Tool '{tool_name}' requires confirmation[/yellow]")
                console.print(f"[dim]Arguments: {arguments}[/dim]")
                response = input("Execute this tool? [y/N]: ").strip().lower()
                confirmed = response in ['y', 'yes']

                # Restart the spinner after confirmation
                # It will continue showing while the tool executes and LLM processes
                if spinner:
                    spinner.start()

                return confirmed

            mcp_client = MCPClient(
                mcp_command,
                notification_handler=notification_handler,
                confirmation_handler=confirmation_handler,
                confirm_prefixes=config.mcp_tool_confirm_prefixes
            )
            await mcp_client.connect()
            console.print("[green]Connected to MCP server[/green]\n")

        # Create chat interface and start interactive session
        chat = ChatInterface(llm_client, config, mcp_client)
        await chat.start()

        # Cleanup
        if mcp_client:
            await mcp_client.disconnect()

        return 0

    except Exception as e:
        logger.error(f"Chat mode failed: {e}")
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        return 1


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = setup_parser()
    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Setup file logging if requested
    if args.log_file:
        # Remove the RichHandler so logs only go to the file
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        file_handler = logging.FileHandler(args.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")

    try:
        # Load configuration
        config = Config.from_env()

        # Override with command-line arguments
        if args.api_key:
            config.gemini_api_key = args.api_key
        if args.granite_url:
            config.granite_url = args.granite_url
        if args.granite_key:
            config.granite_user_key = args.granite_key
        if args.model:
            # Model argument only applies to Gemini
            if args.provider == "gemini" or (config.gemini_api_key and not args.provider):
                config.gemini_model = args.model
            elif args.provider == "granite":
                logger.warning("--model argument is ignored for Granite. Specify full URL in --granite-url or GRANITE_URL")
        if args.namespace:
            config.namespace = args.namespace

        # Determine MCP command
        mcp_command = args.mcp_server or config.mcp_server_command

        # Run interactive chat mode
        return asyncio.run(run_chat_mode(config, mcp_command, args.provider))

    except ValueError as e:
        console = Console()
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("\n[yellow]Please configure an LLM provider in your environment or .env file.[/yellow]")
        console.print("[yellow]For Gemini: export GEMINI_API_KEY='your-api-key-here'[/yellow]")
        console.print("[yellow]For Granite: export GRANITE_URL='...' GRANITE_USER_KEY='...'[/yellow]")
        return 1
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        logger.exception("Unexpected error")
        console = Console()
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
