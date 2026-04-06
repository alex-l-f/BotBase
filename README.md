# BotBase

A modular, expandable chatbot framework built around an LLM agent loop with a plugin-based tool system. Designed as a clean starting point for building any kind of conversational AI agent.

## Project Structure

```
agent.py              Core agent loop — runs the LLM, dispatches tool calls
server.py             Flask API server — exposes chat endpoints

prompts/              Prompt system (profile-based)
  profiles.py         Maps profile names to prompt modules + toolsets
  default.py          Default system prompt (customize this)

tools/                Tool plugin system
  base.py             Abstract BaseTool class — extend this to add tools
  toolsets.py         Named groups of tools for different profiles
  send_message.py     Sends a message back to the user
  finish_turn.py      Ends the current turn, lets the user reply

LMInterface/          LLM backend adapters
  lcpp_interface.py   llama.cpp (via OpenAI-compatible API)
  openai_interface.py OpenAI API
  openrouter_interface.py  OpenRouter API
  vllm_interface.py   vLLM inference server
  mlcllm_interface.py MLC LLM
```

## Quick Start

1. Create a `.env` file with your API key(s):
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

2. Choose your LLM backend by uncommenting the appropriate import in `agent.py`.

3. Run the server:
   ```bash
   python server.py
   ```

4. The API is available at `http://localhost:5551`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/start-chat` | Create a new chat session |
| POST | `/api/chat-profile` | Chat using a named profile |
| POST | `/api/prompt-chat` | Chat with a custom system prompt |
| GET  | `/api/get-messages/<chat_id>` | Poll for new messages |
| GET  | `/api/profiles` | List available profiles |
| POST | `/api/log-event` | Log a custom event |

## Adding Tools

1. Create a new file in `tools/` with a class extending `BaseTool`:
   ```python
   from .base import BaseTool

   class MyTool(BaseTool):
       schema = {
           "type": "function",
           "function": {
               "name": "my_tool",
               "description": "What this tool does",
               "parameters": { ... }
           }
       }

       def execute(self, arguments: dict, context: dict):
           # Your tool logic here
           return "result"
   ```

2. Add the tool name to a toolset in `tools/toolsets.py`.

Tools are auto-discovered on startup — no registration code needed.

## Adding Profiles

1. Create a new prompt file in `prompts/` with a `PROMPT` string.
2. Register it in `prompts/profiles.py` with a name and toolset.

## Customization

- **System prompt**: Edit `prompts/default.py` or create a new profile.
- **LLM backend**: Swap the import in `agent.py` or add a new adapter in `LMInterface/`.
- **Tools**: Drop new tool files into `tools/` and add them to a toolset.
