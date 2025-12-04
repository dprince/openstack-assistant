"""Configuration management for the OpenStack Upgrade Assistant."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for the assistant.

    Attributes:
        gemini_api_key: API key for Google Gemini
        gemini_model: Gemini model to use (default: gemini-2.5-flash)
        granite_url: Full URL for Granite API (including model routing if needed)
        granite_user_key: User key for Granite authentication
        granite_temperature: Temperature for Granite model (default: 0.0)
        granite_max_tokens: Maximum tokens in Granite response
        granite_min_p: Minimum probability threshold for Granite sampling (default: 0.1)
        mcp_server_url: URL of the MCP server
        mcp_server_command: Command to start the MCP server
        mcp_allowed_tools: List of allowed MCP tool names (None = all tools allowed)
        workflow_file: Path to workflow definition file
        system_instruction_file: Path to system instruction file for chat mode
        namespace: Default Kubernetes namespace to use
        mcp_tool_confirm_prefixes: List of tool name prefixes that require user confirmation
    """
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    granite_url: Optional[str] = None
    granite_user_key: Optional[str] = None
    granite_temperature: float = 0.0
    granite_max_tokens: Optional[int] = None
    granite_min_p: float = 0.1
    mcp_server_url: Optional[str] = None
    mcp_server_command: Optional[str] = None
    mcp_allowed_tools: Optional[List[str]] = None
    workflow_file: Optional[Path] = None
    system_instruction_file: Optional[Path] = None
    namespace: Optional[str] = None
    mcp_tool_confirm_prefixes: List[str] = field(default_factory=lambda: ["create_", "watch_", "update_"])

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.

        Looks for a .env file in the current directory or uses
        system environment variables.

        Returns:
            Config: Configured instance

        Raises:
            ValueError: If neither GEMINI_API_KEY nor GRANITE_URL/GRANITE_USER_KEY are set
        """
        # Load .env file if it exists
        load_dotenv()

        # Load Gemini configuration
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # Load Granite configuration
        granite_url = os.getenv("GRANITE_URL")
        granite_user_key = os.getenv("GRANITE_USER_KEY")
        granite_temperature = float(os.getenv("GRANITE_TEMPERATURE", "0.0"))
        granite_max_tokens = None
        if os.getenv("GRANITE_MAX_TOKENS"):
            granite_max_tokens = int(os.getenv("GRANITE_MAX_TOKENS"))
        granite_min_p = float(os.getenv("GRANITE_MIN_P", "0.1"))

        # Validate that at least one LLM provider is configured
        if not gemini_api_key and not (granite_url and granite_user_key):
            raise ValueError(
                "At least one LLM provider must be configured:\n"
                "  - For Gemini: Set GEMINI_API_KEY\n"
                "  - For Granite: Set GRANITE_URL and GRANITE_USER_KEY\n"
                "Set them in your environment or create a .env file."
            )

        # Load common configuration
        mcp_url = os.getenv("MCP_SERVER_URL")
        mcp_command = os.getenv("MCP_SERVER_COMMAND")

        # Load MCP allowed tools
        mcp_allowed_tools = None
        allowed_tools_str = os.getenv("MCP_ALLOWED_TOOLS")
        if allowed_tools_str:
            # Parse comma-separated list
            mcp_allowed_tools = [t.strip() for t in allowed_tools_str.split(",") if t.strip()]

        workflow_file = None
        workflow_path = os.getenv("WORKFLOW_FILE")
        if workflow_path:
            workflow_file = Path(workflow_path)

        system_instruction_file = None
        system_instruction_path = os.getenv("SYSTEM_INSTRUCTION_FILE")
        if system_instruction_path:
            system_instruction_file = Path(system_instruction_path)
        else:
            # Auto-select system instruction file based on configured LLM provider
            # Prefer Granite if configured, otherwise use Gemini (matches provider selection logic)
            base_path = Path(__file__).parent.parent / "system_instructions"

            if granite_url and granite_user_key:
                # Granite is configured - use granite-specific instructions
                granite_path = base_path / "rhoso-upgrade-agent-granite.txt"
                if granite_path.exists():
                    system_instruction_file = granite_path
            elif gemini_api_key:
                # Only Gemini is configured - use gemini-specific instructions
                gemini_path = base_path / "rhoso-upgrade-agent-gemini.txt"
                if gemini_path.exists():
                    system_instruction_file = gemini_path

            # Fallback to generic file if provider-specific file doesn't exist
            if not system_instruction_file:
                default_path = base_path / "rhoso-upgrade-agent.txt"
                if default_path.exists():
                    system_instruction_file = default_path

        # Load namespace
        namespace = os.getenv("NAMESPACE")

        # Load MCP tool confirmation prefixes
        mcp_tool_confirm_prefixes = ["create_", "watch_", "update_"]  # defaults
        prefixes_str = os.getenv("MCP_TOOL_CONFIRM_PREFIXES")
        if prefixes_str:
            # Parse comma-separated list
            mcp_tool_confirm_prefixes = [p.strip() for p in prefixes_str.split(",") if p.strip()]

        return cls(
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            granite_url=granite_url,
            granite_user_key=granite_user_key,
            granite_temperature=granite_temperature,
            granite_max_tokens=granite_max_tokens,
            granite_min_p=granite_min_p,
            mcp_server_url=mcp_url,
            mcp_server_command=mcp_command,
            mcp_allowed_tools=mcp_allowed_tools,
            workflow_file=workflow_file,
            system_instruction_file=system_instruction_file,
            namespace=namespace,
            mcp_tool_confirm_prefixes=mcp_tool_confirm_prefixes,
        )
