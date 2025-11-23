# OpenStack Assistant

A simplified command-line chat assistant for OpenStack, powered by AI (Google Gemini or IBM Granite) and Model Context Protocol (MCP) servers.

## Features

- **Interactive Chat Interface**: Chat with AI (IBM Granite or Google Gemini) about OpenStack
- **Multiple LLM Providers**: Support for Google Gemini and IBM Granite models
- **MCP Server Integration**: Connect to MCP servers to execute tools and access external data
- **Workflow Automation**: Define and execute multi-step workflows combining AI queries and tool calls
- **Simple Configuration**: Easy setup with environment variables

## Installation

### Prerequisites

- Python 3.9 or higher
- One of the following AI providers:
  - A Google Gemini API key (get one at https://makersuite.google.com/app/apikey)
  - An IBM Granite API endpoint URL and user key
- Optional: Node.js (for MCP servers like filesystem, etc.)

### Install from source

```bash
cd openstack-assistant
pip install -e .
```

## Configuration

Create a `.env` file in your working directory or set environment variables:

```bash
# ============================================================================
# LLM Provider Configuration (Choose ONE)
# ============================================================================

# Option 1: Google Gemini
GEMINI_API_KEY=your-api-key-here

# Optional: Gemini model (default: gemini-2.5-flash)
# Options: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, gemini-2.0-flash-lite
GEMINI_MODEL=gemini-2.5-flash

# Option 2: IBM Granite
# Requires both URL and User Key
GRANITE_URL=https://your-granite-api-endpoint.com
GRANITE_USER_KEY=your-granite-user-key-here

# Optional: Granite temperature (default: 0.0)
GRANITE_TEMPERATURE=0.1

# ============================================================================
# MCP and Workflow Configuration
# ============================================================================

# Optional: MCP server command
MCP_SERVER_COMMAND="npx @modelcontextprotocol/server-filesystem /tmp"

# Optional: Default workflow file
WORKFLOW_FILE=/path/to/workflow.json

# Optional: System instruction file for chat mode
SYSTEM_INSTRUCTION_FILE=/path/to/system-instruction.txt
```

See `.env.example` for a complete example.

## Usage

### Interactive Chat Mode

Start an interactive chat session:

```bash
openstack-assistant
```

or explicitly:

```bash
openstack-assistant --interactive
```

In interactive mode, you can use these commands:
- `.exit` or `.quit` - Exit the chat
- `.clear` - Clear conversation history
- `.tools` - Show available MCP tools (if connected)
- `.help` - Show help message

### Single Message Mode

Send a single message and exit:

```bash
openstack-assistant -m "How do I upgrade from OpenStack Wallaby to Xena?"
```

### With MCP Server

Connect to an MCP server for tool access:

```bash
openstack-assistant --mcp-server "npx @modelcontextprotocol/server-filesystem /tmp"
```

### Workflow Mode

Execute a predefined workflow:

```bash
openstack-assistant --workflow workflow.json
```

See `examples/openstack-upgrade-workflow.json` for an example workflow definition.

### Command-line Options

```
usage: openstack-assistant [-h] [--version] [-v] [-q] [-i] [-m MESSAGE]
                          [--mcp-server MCP_SERVER] [-w WORKFLOW]
                          [--api-key API_KEY] [--model MODEL]
                          [--granite-url GRANITE_URL]
                          [--granite-user-key GRANITE_USER_KEY]

OpenStack Assistant - AI-powered chat interface for OpenStack

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v, --verbose         Enable verbose logging
  -q, --quiet           Suppress non-error output
  -i, --interactive     Start interactive chat session (default if no message
                        provided)
  -m MESSAGE, --message MESSAGE
                        Send a single message and exit
  --mcp-server MCP_SERVER
                        Command to start MCP server (e.g., 'npx
                        @modelcontextprotocol/server-filesystem /path')
  -w WORKFLOW, --workflow WORKFLOW
                        Path to workflow JSON file to execute
  --api-key API_KEY     Gemini API key (overrides GEMINI_API_KEY environment
                        variable)
  --model MODEL         Gemini model to use (default: gemini-2.5-flash)
  --granite-url GRANITE_URL
                        Granite API URL (overrides GRANITE_URL environment
                        variable)
  --granite-user-key GRANITE_USER_KEY
                        Granite user key (overrides GRANITE_USER_KEY
                        environment variable)
```

## Workflow Files

Workflows are defined in JSON format. Here's an example:

```json
{
  "name": "OpenStack Version Check",
  "description": "Check OpenStack version and get upgrade advice",
  "system_instruction": "You are an OpenStack upgrade specialist. Only provide information related to OpenStack upgrades and follow the workflow steps precisely.",
  "steps": [
    {
      "name": "Check Version",
      "type": "tool",
      "content": "read_file",
      "arguments": {
        "path": "/etc/openstack-release"
      },
      "description": "Read the OpenStack release file"
    },
    {
      "name": "Get Upgrade Advice",
      "type": "ask",
      "content": "Based on the OpenStack release information, what are the recommended upgrade steps?",
      "description": "Ask AI for upgrade recommendations"
    }
  ]
}
```

### Workflow Configuration

- **name**: Name of the workflow
- **description**: Description of what the workflow does
- **system_instruction** (optional): System-level instructions that define the agent's identity, behavior, and constraints. This scopes the AI to specific tasks and prevents it from deviating from the intended purpose.
- **steps**: Array of steps to execute

### Workflow Step Types

- **ask**: Send a question to Gemini AI
  - `content`: The question to ask
- **tool**: Call an MCP tool
  - `content`: Tool name
  - `arguments`: Dictionary of tool arguments

### Agent Identity and Scoping

The `system_instruction` field allows you to define a specific identity for the AI agent. This is particularly useful for constraining the agent to specific tasks like OpenStack RHOSO cluster upgrades. The system instruction:

- Defines what the agent is responsible for
- Sets clear boundaries on what operations are allowed
- Specifies constraints and rules the agent must follow
- Can allow slight deviations (like retrying failed steps) while maintaining overall scope

See `examples/rhoso-upgrade-workflow.json` for a comprehensive example of an agent scoped to RHOSO cluster upgrades.

## Agent Identity in Chat Mode

In addition to workflows, you can configure the AI agent's identity and behavior in interactive chat mode by providing a system instruction file. This allows you to scope the agent to specific tasks (like RHOSO cluster upgrades) without using the workflow feature.

### Configuring System Instructions for Chat

Set the `SYSTEM_INSTRUCTION_FILE` environment variable to point to a text file containing your system instructions:

```bash
export SYSTEM_INSTRUCTION_FILE=/path/to/rhoso-upgrade-agent.txt
openstack-assistant
```

Or add it to your `.env` file:

```bash
SYSTEM_INSTRUCTION_FILE=examples/rhoso-upgrade-agent.txt
```

### System Instruction Format

System instruction files are plain text files that define:
- The agent's role and identity
- Scope and constraints on what the agent should do
- Rules and procedures the agent must follow
- Allowed deviations (like retrying failed operations or looping)

Example system instruction file (`examples/rhoso-upgrade-agent.txt`):

```text
You are an OpenStack RHOSO Cluster Upgrade Assistant.

Your primary responsibility is to guide and assist with RHOSO cluster
upgrades following established Red Hat procedures and best practices.

## Constraints and Rules

1. Stay Focused: Only provide assistance related to RHOSO cluster upgrades
2. Follow Established Procedures: Always reference Red Hat documentation
3. Safety First: Always recommend backups before upgrades
4. Controlled Deviations: You may retry failed steps with troubleshooting
...
```

### Benefits of Chat Mode System Instructions

- **Focused Agent**: Keeps the AI focused on specific tasks (e.g., RHOSO upgrades)
- **Consistent Behavior**: Ensures the agent follows your defined procedures
- **Flexible Interaction**: Unlike workflows, you can have freeform conversations while maintaining scope
- **Easy Configuration**: Simple text file, no JSON structure required
- **Reusable**: Same instruction file can be used across multiple chat sessions

### Difference from Workflows

- **Workflows**: Define explicit step-by-step procedures with AI queries and tool calls. Best for repeatable, automated tasks.
- **Chat Mode System Instructions**: Define the agent's identity and constraints for freeform interactive conversations. Best for guided assistance and troubleshooting.

You can use both together: workflows for automation, and system instructions for interactive support.

## MCP Servers

The assistant can connect to any MCP server that follows the Model Context Protocol. Some useful servers:

- **filesystem**: Access local files
  ```bash
  npx @modelcontextprotocol/server-filesystem /path/to/directory
  ```

- **sqlite**: Query SQLite databases
  ```bash
  npx @modelcontextprotocol/server-sqlite /path/to/database.db
  ```

- **Custom servers**: You can create your own MCP servers

## Examples

### Example 1: Basic Chat with Gemini

```bash
export GEMINI_API_KEY=your-key-here
openstack-assistant
```

Then ask questions like:
- "What are the prerequisites for upgrading from Wallaby to Xena?"
- "How do I backup my OpenStack database before upgrading?"
- "What's the recommended order for upgrading OpenStack services?"

### Example 1b: Basic Chat with Granite

```bash
export GRANITE_URL=https://your-granite-api-endpoint.com
export GRANITE_USER_KEY=your-granite-user-key-here
openstack-assistant
```

Or using command-line options:

```bash
openstack-assistant --granite-url https://your-granite-api-endpoint.com \
                   --granite-user-key your-granite-user-key-here
```

### Example 2: With Filesystem Access

```bash
openstack-assistant --mcp-server "npx @modelcontextprotocol/server-filesystem /etc/openstack"
```

The AI can now access files in `/etc/openstack` through MCP tools.

### Example 3: Run a Workflow

```bash
openstack-assistant --workflow examples/openstack-upgrade-workflow.json
```

### Example 4: Chat with RHOSO Upgrade Agent Identity

```bash
export SYSTEM_INSTRUCTION_FILE=examples/rhoso-upgrade-agent.txt
openstack-assistant
```

The AI will now act as a specialized RHOSO upgrade assistant, staying focused on upgrade-related tasks.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd openstack-assistant

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

## Architecture

The assistant consists of several components:

- **Config** (`config.py`): Configuration management using environment variables
- **GeminiClient** (`gemini_client.py`): Google Gemini API integration
- **MCPClient** (`mcp_client.py`): Model Context Protocol client for tool access
- **ChatInterface** (`chat.py`): Interactive chat UI using Rich and prompt_toolkit
- **Workflow** (`workflow.py`): Workflow definition and execution engine
- **CLI** (`cli.py`): Command-line interface and main entry point

## Comparison with command-line-assistant

This is a simplified version of the RHEL command-line-assistant project:

**Simplified aspects:**
- No daemon architecture (runs as a single process)
- No DBus communication
- No database for history (in-memory only)
- Simplified configuration (environment variables only)
- Direct Gemini API integration (vs backend service)

**New features:**
- MCP server integration for tool access
- Workflow automation system
- Focused on OpenStack upgrade use cases

## Troubleshooting

### API Key Issues

**For Gemini:**
If you get authentication errors:
1. Verify your API key is correct
2. Check that the environment variable is set: `echo $GEMINI_API_KEY`
3. Try setting it directly: `openstack-assistant --api-key your-key-here`

**For Granite:**
If you get authentication errors:
1. Verify your Granite URL and user key are correct
2. Check that the environment variables are set:
   ```bash
   echo $GRANITE_URL
   echo $GRANITE_USER_KEY
   ```
3. Try setting them directly:
   ```bash
   openstack-assistant --granite-url https://your-endpoint.com \
                      --granite-user-key your-key-here
   ```

### MCP Connection Issues

If MCP server connection fails:
1. Verify the server command is correct
2. Check that Node.js is installed (for npx-based servers)
3. Try running the server command manually first
4. Check the logs with `--verbose` flag

### Rate Limiting

**For Gemini:**
If you hit Gemini API rate limits:
- Wait a few minutes before retrying
- Consider using a different model with `--model`
- Check your API quota at Google AI Studio

**For Granite:**
If you encounter rate limits or quota issues:
- Wait a few minutes before retrying
- Contact your Granite API administrator for quota adjustments

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or support, please open an issue on the repository.
