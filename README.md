# BotBase

A modular, expandable chatbot framework built around an LLM agent loop with a plugin-based tool system. Designed as a clean starting point for building any kind of conversational AI agent.

Ships with two selectable agent architectures (see [Multi-Agent Architecture](#multi-agent-architecture)):

- **`multi`** (default) — a supervisor-topology system: a user-facing **coach** that delegates library research to a read-only **summarizer** sub-agent and recalls past sessions from an episodic **memory** store.
- **`single`** — the original one-agent librarian, kept as the benchmark baseline.

## Project Structure

```
agent.py              Core agent loop — runs the coach LLM, dispatches tool calls
summarizer_agent.py   Summarizer sub-agent — read-only research loop with a
                      structured output contract (multi arch only)
memory.py             Episodic memory store — SQLite + FTS5, single-writer
trace.py              Per-agent instrumentation — streams events to the debug UI
                      and persists them to the trace log
server.py             Flask API server — exposes chat endpoints
embedding_service.py  FastAPI microservice — hybrid HNSW + BM25 search
embedding_client.py   Client for the embedding service (used by tools)
import_resources.py   Importer script — converts .txt/.json into searchable resources
import_transcripts.py Importer for per-topic transcript JSONs (data/Transcripts/)
import_elearning.py   Indexes e-learning course pages into the topic providers

prompts/              Prompt system (profile-based)
  profiles.py         Maps profile names to prompt modules + toolsets
  default.py          Default system prompt (customize this)
  summarizer.py       System prompt for the summarizer sub-agent
  coach_overlay.py    Multi-agent overlay appended to topic prompts

tools/                Tool plugin system
  base.py             Abstract BaseTool class — extend this to add tools
  toolsets.py         Named groups of tools for different profiles
  send_message.py     Sends a message back to the user
  finish_turn.py      Ends the current turn, lets the user reply
  search_resources.py Searches the resource database via embeddings
  examine_resource.py Returns full details for a specific resource
  provide_file.py     Sends a resource file to the user (player / download card)
  open_course_page.py Opens an e-learning course page in the viewer panel
  switch_mode.py      Switches the active topic mode + resource library
  ask_library.py      Coach → summarizer delegation (multi arch)
  memory_search.py    Coach-side recall over past-session logs (multi arch)
  return_summary.py   Summarizer's structured output contract (multi arch)

LMInterface/          LLM backend adapters
  lcpp_interface.py   llama.cpp (via OpenAI-compatible API)
  openai_interface.py OpenAI API
  openrouter_interface.py  OpenRouter API
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
| GET  | `/memory` | Memory browser/editor page |
| GET  | `/api/memory/stats` | Episode/session counts and known modes |
| GET  | `/api/memory/snapshot` | The MEMORY SNAPSHOT block as it would be injected now |
| GET/POST | `/api/memory/episodes` | Browse (filter/paginate) or add memory entries |
| PUT/DELETE | `/api/memory/episodes/<id>` | Edit or delete one memory entry |
| GET/POST | `/api/memory/scenarios` | List scenarios / save current memory as one |
| POST | `/api/memory/scenarios/<slug>/load` | Replace memory with a scenario (optional re-dating) |
| DELETE | `/api/memory/scenarios/<slug>` | Delete a saved scenario file |

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

## E-learning Course Integration

The chatbot can search the pages of a SCORM e-learning module (an Articulate
Rise export) and open them beside the chat.

**Serving the course.** Drop the SCORM export directory (the folder holding
`imsmanifest.xml`) into the project root. The server auto-discovers it,
serves it under `/scorm/<package>/...`, and `/api/scorm/index` feeds the
frontend's "Modules" panel with deep links to every lesson.

**Indexing the pages.** Per-lesson metadata lives in
`Elearningcourse/<section>/<lesson>/rag_content.json` (lesson id, title,
section content, and an LLM-generated summary/keywords/takeaways block).
Import it with:

```bash
python import_elearning.py            # add --dry-run to preview the mapping
```

The script maps each section folder onto one of the topic providers (see
`SECTION_TOPIC_RULES` in the script), validates every lesson id against the
Rise course, and appends the pages to the topic's `database.db` and HNSW
index as `source_type: "course_page"` resources. It is safe to re-run; a
re-import replaces the previous course pages. Restart the embedding service
afterwards so it picks up the updated indexes.

**Sending pages to the user.** Course pages come back from
`search_resources` like any other resource. The bot calls the
`open_course_page` tool with the resource id; the frontend then renders a
card in the chat and opens the lesson in the content viewer next to the
conversation (`provide_file` stays reserved for actual files).

## Multi-Agent Architecture

The design follows the prototype path in `multi-agent-paradigms-2026.md` §6: a supervisor topology one level deep, split along **context boundaries** rather than job titles.

**Coach (orchestrator, user-facing).** The existing topic-bot loop re-prompted with a multi-agent overlay. It owns the conversation and every user-facing action (`send_message`, `provide_file`, `open_course_page`, `switch_mode`, `finish_turn`) — writes stay single-threaded. Instead of searching the library itself, it delegates via two tools:

- **`ask_library(question)`** — spawns a fresh **summarizer** run and returns its structured summary.
- **`memory_search(query)`** — keyword search over past-session logs.

**Summarizer (read-only explorer).** A second instance of the same loop machinery with its own context window and a read-only toolset (`search_resources`, `examine_resource`). It must finish by calling `return_summary`, which enforces a fixed output contract — `answer`, `key_points`, `resources` (ids validated against what the run actually retrieved), `confidence`, `notes` — stamped with `source` + `timestamp`. Retrieved resources are merged back into the coach's context so the cited ids resolve in `provide_file` / `open_course_page`. Effort is capped (8 tool calls) with scaling rules in the prompt. Set `SUMMARIZER_MODEL` in `.env` to run it on a cheaper model.

**Memory (not an agent).** A single-writer SQLite + FTS5 store (`memory.db`):

- *Write path* — plain episodic logging of each completed turn, from exactly one call site in `agent.py`. No LLM, no extraction.
- *Read path* — the `memory_search` tool (pull, on the coach's initiative).
- *Always-injected* — one hard-capped (~300 token) MEMORY SNAPSHOT block derived with plain SQL: prior session count, topics visited, recent asks.
- *Instrumentation* — every tool call and agent event is persisted to `trace_log` (the future push-channel spec and eval corpus, per the brief).
- *Browser/editor* — `/memory` (also via the 🧠 Memory button in debug mode) shows the live injected snapshot and the episodic log with search, filters, inline editing, deletion, and an add form — useful for staging demo state and for showing an audience exactly what the bot remembers. Edits go through `MemoryStore`'s admin methods, preserving the single-writer rule. Note the store is global: all sessions (including simulator runs) share one memory pool.
- *Scenarios* — the browser's Scenarios card snapshots the whole episodic log under a name and reloads it later (destructive replace, confirm-guarded), so a staged demo ("user returns the day after a session") is repeatable. On load, timestamps can be re-dated so the newest entry is always N days ago, preserving relative spacing — the "day after" framing survives no matter when the demo runs. Scenarios are plain JSON files under `scenarios/`; sync or hand-edit them freely. After loading one, hit Reset in the chat UI so the new state applies to a fresh session.

**Selecting the architecture.** Server default via `--arch multi|single` or `BOTBASE_ARCH`; the frontend's debug mode has a per-request selector, so you can run the brief's §6 comparison — same question, same budget, single vs. multi — from the UI.

**Debug sub-panes.** The Debug toggle now also reveals a bottom panel with one live sub-pane per agent — **Coach**, **Summarizer**, **Memory** — streaming each agent's model output, tool calls/results, spawns, structured summaries, and memory reads/writes as they happen.

Deliberately deferred (per the brief): proactive push channels, extraction/conflict detection on memory writes, semantic/procedural memory tiers, a verification agent, and vector indexing of the logs.

## Customization

- **System prompt**: Edit `prompts/default.py` or create a new profile.
- **LLM backend**: Swap the import in `agent.py` or add a new adapter in `LMInterface/`.
- **Tools**: Drop new tool files into `tools/` and add them to a toolset.
