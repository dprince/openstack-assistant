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
from .mcp_client import MCPClient
from .workflow import Workflow, WorkflowRunner

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


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

    # Chat mode arguments
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive chat session (default if no message provided)",
    )

    parser.add_argument(
        "-m",
        "--message",
        type=str,
        help="Send a single message and exit",
    )

    # MCP arguments
    parser.add_argument(
        "--mcp-server",
        type=str,
        help="Command to start MCP server (e.g., 'npx @modelcontextprotocol/server-filesystem /path')",
    )

    # Workflow arguments
    parser.add_argument(
        "-w",
        "--workflow",
        type=Path,
        help="Path to workflow JSON file to execute",
    )

    # Configuration
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (overrides GEMINI_API_KEY environment variable)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )

    return parser


async def run_workflow_mode(
    config: Config,
    workflow_path: Path,
    mcp_command: Optional[str] = None,
) -> int:
    """Run workflow mode.

    Args:
        config: Configuration object
        workflow_path: Path to workflow file
        mcp_command: Optional MCP server command

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    console = Console()

    try:
        # Load workflow
        workflow = Workflow.from_file(workflow_path)
        console.print(f"[cyan]Loading workflow: {workflow.name}[/cyan]")
        console.print(f"[dim]{workflow.description}[/dim]\n")

        # Initialize clients
        gemini_client = GeminiClient(config)

        mcp_client = None
        if mcp_command:
            console.print(f"[cyan]Connecting to MCP server...[/cyan]")
            mcp_client = MCPClient(mcp_command)
            await mcp_client.connect()
            console.print(f"[green]Connected to MCP server[/green]\n")

        # Run workflow
        runner = WorkflowRunner(gemini_client, mcp_client)
        results = await runner.run_workflow(workflow)

        # Display results
        console.print("\n[green]Workflow completed successfully![/green]\n")
        console.print("[cyan]Results:[/cyan]")
        for i, result in enumerate(results, 1):
            console.print(f"\n{i}. {result['step']} ({result['type']})")
            if "error" in result:
                console.print(f"   [red]Error: {result['error']}[/red]")
            else:
                console.print(f"   [green]Result: {result['result']}[/green]")

        # Cleanup
        if mcp_client:
            await mcp_client.disconnect()

        return 0

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        return 1


async def run_chat_mode(
    config: Config,
    interactive: bool = True,
    message: Optional[str] = None,
    mcp_command: Optional[str] = None,
) -> int:
    """Run chat mode.

    Args:
        config: Configuration object
        interactive: Whether to run in interactive mode
        message: Optional single message to send
        mcp_command: Optional MCP server command

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Initialize Gemini client
        gemini_client = GeminiClient(config)

        # Initialize MCP client if provided
        mcp_client = None
        if mcp_command:
            console = Console()
            console.print("[cyan]Connecting to MCP server...[/cyan]")
            mcp_client = MCPClient(mcp_command)
            await mcp_client.connect()
            console.print("[green]Connected to MCP server[/green]\n")

        # Create chat interface
        chat = ChatInterface(gemini_client, config, mcp_client)

        # Run in appropriate mode
        if message:
            # Single message mode
            chat.send_single_message(message)
        else:
            # Interactive mode
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

    try:
        # Load configuration
        config = Config.from_env()

        # Override with command-line arguments
        if args.api_key:
            config.gemini_api_key = args.api_key
        if args.model:
            config.gemini_model = args.model

        # Determine MCP command
        mcp_command = args.mcp_server or config.mcp_server_command

        # Run appropriate mode
        if args.workflow:
            # Workflow mode
            return asyncio.run(run_workflow_mode(config, args.workflow, mcp_command))
        else:
            # Chat mode (interactive or single message)
            interactive = args.interactive or not args.message
            return asyncio.run(run_chat_mode(config, interactive, args.message, mcp_command))

    except ValueError as e:
        console = Console()
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("\n[yellow]Please set the GEMINI_API_KEY environment variable or create a .env file.[/yellow]")
        console.print("[yellow]Example: export GEMINI_API_KEY='your-api-key-here'[/yellow]")
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
