"""Configuration management for the OpenStack Upgrade Assistant."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for the assistant.

    Attributes:
        gemini_api_key: API key for Google Gemini
        gemini_model: Gemini model to use (default: gemini-2.5-flash)
        mcp_server_url: URL of the MCP server
        mcp_server_command: Command to start the MCP server
        workflow_file: Path to workflow definition file
        system_instruction_file: Path to system instruction file for chat mode
    """
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    mcp_server_url: Optional[str] = None
    mcp_server_command: Optional[str] = None
    workflow_file: Optional[Path] = None
    system_instruction_file: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.

        Looks for a .env file in the current directory or uses
        system environment variables.

        Returns:
            Config: Configured instance

        Raises:
            ValueError: If GEMINI_API_KEY is not set
        """
        # Load .env file if it exists
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Set it in your environment or create a .env file."
            )

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        mcp_url = os.getenv("MCP_SERVER_URL")
        mcp_command = os.getenv("MCP_SERVER_COMMAND")

        workflow_file = None
        workflow_path = os.getenv("WORKFLOW_FILE")
        if workflow_path:
            workflow_file = Path(workflow_path)

        system_instruction_file = None
        system_instruction_path = os.getenv("SYSTEM_INSTRUCTION_FILE")
        if system_instruction_path:
            system_instruction_file = Path(system_instruction_path)
        else:
            # Default to rhoso-upgrade-agent.txt if it exists
            default_path = Path(__file__).parent.parent / "system_instructions" / "rhoso-upgrade-agent.txt"
            if default_path.exists():
                system_instruction_file = default_path

        return cls(
            gemini_api_key=api_key,
            gemini_model=model,
            mcp_server_url=mcp_url,
            mcp_server_command=mcp_command,
            workflow_file=workflow_file,
            system_instruction_file=system_instruction_file,
        )
