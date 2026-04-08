# BotBase

A modular, expandable chatbot framework built around an LLM agent loop with a plugin-based tool system. Designed as a clean starting point for building any kind of conversational AI agent.

## Project Structure

```
agent.py              Core agent loop — runs the LLM, dispatches tool calls
server.py             Flask API server — exposes chat endpoints
embedding_service.py  FastAPI microservice — hybrid HNSW + BM25 search
embedding_client.py   Client for the embedding service (used by tools)
import_resources.py   Importer script — converts .txt/.json into searchable resources

prompts/              Prompt system (profile-based)
  profiles.py         Maps profile names to prompt modules + toolsets
  default.py          Default system prompt (customize this)

tools/                Tool plugin system
  base.py             Abstract BaseTool class — extend this to add tools
  toolsets.py         Named groups of tools for different profiles
  send_message.py     Sends a message back to the user
  finish_turn.py      Ends the current turn, lets the user reply
  search_resources.py Searches the resource database via embeddings
  examine_resource.py Returns full details for a specific resource

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

3. Import your resources (requires `sentence-transformers`, `hnswlib`, `torch`):
   ```bash
   python import_resources.py ./your_resources/ --output ./processed_resources/your_provider
   ```
   This reads `.txt` and `.json` files from the input directory, embeds them, and writes the index and database to the output directory. See [Resource Search](#resource-search) for input format details.

4. Start the embedding service (must be running before the chatbot server):
   ```bash
   uvicorn embedding_service:app --host 0.0.0.0 --port 8200 --workers 1
   ```
   Make sure your output directory from step 3 is listed in `PROVIDER_DIRS` in `embedding_service.py`.

5. Start the chatbot server:
   ```bash
   python server.py
   ```

6. Open `http://localhost:5551` in your browser.

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

## Resource Search

The framework includes a hybrid search system for retrieving resources from an embedded database.

### Importing Resources

Use `import_resources.py` to convert a directory of `.txt` and `.json` files into the format the embedding service expects:

```bash
python import_resources.py ./my_resources/ --output ./processed_resources/my_provider
```

**Supported input formats:**

- **`.json`** — A single object or array of objects with `title` and `description` fields. Optional: `physical_address`, `portal_url`, `latitude`, `longitude`.
- **`.txt`** — One resource per file. The filename becomes the title; file contents become the description.

The script produces `database.db`, `embeddings.bin`, `embedded_texts.pkl`, and `text_to_resource_mapping.pkl` in the output directory.

### Running the Embedding Service

The embedding service must be running for both importing and searching:

```bash
uvicorn embedding_service:app --host 0.0.0.0 --port 8200 --workers 1
```

Add your output directory to `PROVIDER_DIRS` in `embedding_service.py` so it loads on startup.

### Search Tools

Both tools are included in the default toolset:

- **`search_resources`** — Searches the resource database with a query string and returns matching results.
- **`examine_resource`** — Returns full details for a specific resource by ID.

## Customization

- **System prompt**: Edit `prompts/default.py` or create a new profile.
- **LLM backend**: Swap the import in `agent.py` or add a new adapter in `LMInterface/`.
- **Tools**: Drop new tool files into `tools/` and add them to a toolset.
