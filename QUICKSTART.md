# Quick Start Guide

Get started with OpenStack Upgrade Assistant in 5 minutes!

## 1. Install

```bash
cd openstack-upgrade-assistant
pip install -e .
```

## 2. Configure

Create a `.env` file with your Gemini API key:

```bash
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

Get your API key from: https://makersuite.google.com/app/apikey

## 3. Run

### Interactive Chat

```bash
openstack-assistant
```

Then ask questions like:
- "How do I upgrade OpenStack from Wallaby to Xena?"
- "What pre-upgrade checks should I perform?"
- "How do I backup my OpenStack database?"

### Single Question

```bash
openstack-assistant -m "What are the OpenStack core services?"
```

### Run a Workflow

```bash
openstack-assistant -w examples/simple-workflow.json
```

## 4. Advanced Usage

### With MCP Server (File Access)

First install the MCP filesystem server:

```bash
npm install -g @modelcontextprotocol/server-filesystem
```

Then run with file access:

```bash
openstack-assistant --mcp-server "npx @modelcontextprotocol/server-filesystem /etc/openstack"
```

Now the AI can read your OpenStack configuration files!

### Custom Workflow

Create your own `my-workflow.json`:

```json
{
  "name": "My Custom Workflow",
  "description": "Custom OpenStack checks",
  "steps": [
    {
      "name": "Ask About Nova",
      "type": "ask",
      "content": "What is Nova in OpenStack?",
      "description": "Learn about Nova"
    },
    {
      "name": "Ask About Neutron",
      "type": "ask",
      "content": "What is Neutron in OpenStack?",
      "description": "Learn about Neutron"
    }
  ]
}
```

Run it:

```bash
openstack-assistant -w my-workflow.json
```

## Interactive Commands

When in interactive mode (just run `openstack-assistant`), you can use:

- `.help` - Show help
- `.clear` - Clear conversation history
- `.tools` - Show available MCP tools (if connected to MCP server)
- `.exit` or `.quit` - Exit the chat

## Troubleshooting

### "GEMINI_API_KEY is required"

Make sure you created a `.env` file with your API key:

```bash
echo "GEMINI_API_KEY=your-key" > .env
```

Or export it:

```bash
export GEMINI_API_KEY=your-key
```

### MCP Server Issues

If using MCP servers, make sure Node.js is installed:

```bash
node --version  # Should show v18 or higher
```

Install the MCP server you want to use:

```bash
npm install -g @modelcontextprotocol/server-filesystem
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out the example workflows in the `examples/` directory
- Learn about MCP servers: https://modelcontextprotocol.io/

## Example Session

```
$ openstack-assistant
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                  Welcome                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

OpenStack Upgrade Assistant

Welcome to the interactive chat interface!

>>> What are the main steps to upgrade OpenStack?

[Thinking...]