### Agents

This repo can be worked on with multiple LLM “agents”. Use whichever you prefer; the MCP server (`mcp.py`) exposes a single `run` tool/route to execute the `main.py` workflow.

- **Claude**: see Anthropic Claude docs at `https://www.anthropic.com/claude`
- **Gemini**: see Google Gemini docs at `https://ai.google.dev/gemini-api`
- **OpenAI**: see OpenAI platform docs at `https://platform.openai.com/docs`

### MCP entrypoint

- **Server**: `mcp.py`
- **Single route**: `POST /run` (also available as MCP tool `run`)
- **Status**: `GET /status`

