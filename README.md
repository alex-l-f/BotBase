# BotBase

A modular, expandable chatbot framework built on Google's [Agent Development Kit](https://google.github.io/adk-docs/) (ADK) with a plugin-based tool system. Designed as a clean starting point for building any kind of conversational AI agent.

Ships with two selectable agent architectures (see [Multi-Agent Architecture](#multi-agent-architecture)):

- **`multi`** (default) — a supervisor-topology system: a user-facing **coach** that delegates library research to a read-only **summarizer** sub-agent and recalls past sessions from a profile-scoped, privacy-preserving **template memory** store written by a **memory agent**.
- **`single`** — the original one-agent librarian, kept as the benchmark baseline.

## Project Structure

```
adk_app/              The agents, on ADK
  runtime.py          One coach turn: ADK session per chat, Runner, history
                      export, memory extraction — the API server.py calls
  agents.py           The coach / summarizer / memory LlmAgents + runners
  subagents.py        Summarizer and memory-agent sub-runs (fresh session
                      per spawn, terminal-tool contract, one nudge)
  tools.py            Adapts the tools/ plugins to ADK BaseTool; ask_library
                      as an ADK sub-run
  turns.py            Per-turn context registry the plugins read from
  tracing.py          ADK plugin that streams every model/tool event to the
                      debug UI and trace log
  history.py          ADK events <-> the message list the frontend renders
  models.py           LLM backends (OpenRouter / llama.cpp via LiteLLM, Gemini)
memory.py             Profile-scoped template memory store — SQLite + FTS5,
                      single-writer
memory_templates.py   Template registry — the fixed vocabulary of what memory
                      is allowed to store (the privacy contract)
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
  memory_agent.py     System prompt for the memory extraction sub-agent
  coach_overlay.py    Multi-agent adaptation of the topic prompts: rewrites
                      their search/examine guidance onto ask_library and
                      appends the coach overlay
  tone.py             House tone (neutral but serious, no emojis) appended
                      to the router and every topic prompt

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
  memory_search.py    Coach-side recall over the profile's memory notes (multi arch)
  return_summary.py   Summarizer's structured output contract (multi arch)
  record_memories.py  Memory agent's structured output contract — only admits
                      registered templates (multi arch)

LMInterface/          Standalone OpenAI-compatible client adapters from the
                      pre-ADK loop; no longer used by the server
```

## Quick Start

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API key(s):
   ```
   OPENROUTER_API_KEY=your_key_here
   ```
   Pick the LLM backend with `python server.py --backend openrouter|llama_cpp|gemini`
   (or `BOTBASE_BACKEND`). `openrouter` uses `BOTBASE_MODEL` (default
   `openai/gpt-oss-20b`) with `BOTBASE_REASONING_EFFORT` (`low` / `medium` /
   `high` / `off`, default `medium`; `SUMMARIZER_REASONING_EFFORT` and
   `MEMORY_REASONING_EFFORT` override it for the helper agents — set them to
   `low`, the helpers' output is structured and thinking tokens are what a
   turn's seconds are made of). `BOTBASE_MAX_OUTPUT_TOKENS` (default 8192)
   caps one model reply including thinking; `BOTBASE_LLM_RETRIES` (default 4)
   retries rate limits and transient errors with backoff. The summarizer's
   read budget is 4 searches + 3 examines per run, each examine capped at
   `SUMMARIZER_EXAMINE_CHARS` (default 12000). `llama_cpp` talks to `LLAMA_CPP_URL` (default
   `http://localhost:8080/v1`); `gemini` needs `GOOGLE_API_KEY` and uses
   `GEMINI_MODEL` (default `gemini-2.5-flash`).

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
| POST | `/api/chat-profile` | Chat using a named profile (accepts `user_id` for memory scope) |
| POST | `/api/prompt-chat` | Chat with a custom system prompt |
| GET  | `/api/get-messages/<chat_id>` | Poll for new messages |
| GET  | `/api/profiles` | List available prompt profiles |
| GET/POST | `/api/user-profiles` | List / create user profiles (memory scope) |
| DELETE | `/api/user-profiles/<id>` | Delete a user profile and all its memory notes |
| POST | `/api/log-event` | Log a custom event |
| GET  | `/memory` | Memory browser/editor page (`?user=<id>` preselects a profile) |
| GET  | `/api/memory/stats` | Profile and note counts |
| GET  | `/api/memory/templates` | The memory template registry (slots + enums) |
| GET  | `/api/memory/snapshot?user_id=` | The MEMORY SNAPSHOT block as it would be injected now |
| GET/POST | `/api/memory/memories?user_id=` | Browse (filter/paginate) or add template notes |
| PUT/DELETE | `/api/memory/memories/<id>` | Edit slot values / delete one note |
| GET/POST | `/api/memory/scenarios` | List scenarios / save a profile's notes as one |
| POST | `/api/memory/scenarios/<slug>/load` | Replace a profile's notes with a scenario (optional re-dating) |
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

**PDFs are never sent.** `provide_file` refuses `pdf` resources
(`UNSERVED_SOURCE_TYPES` in `tools/provide_file.py`) and tells the bot to
have the content read instead: in the multi architecture the coach asks
the library summarizer (`ask_library`), which examines the resource's
`full_transcript` and returns a summary the coach relays; in the single
architecture the bot calls `examine_resource` itself. Summaries flag such
resources with `deliverable: false`. Audio, video and docx/pptx files are
still delivered as before.

## Multi-Agent Architecture

The design follows the prototype path in `multi-agent-paradigms-2026.md` §6: a supervisor topology one level deep, split along **context boundaries** rather than job titles.

**On ADK.** Each agent is an ADK `LlmAgent` (`adk_app/agents.py`); the framework owns the model loop, function calling, sessions and callbacks:

- A chat is an ADK session (session id = chat id, `InMemorySessionService`), so history lives server-side. The client's `fullContext` is only used to seed a session the server has never seen. `adk_app/history.py` renders the session's events back into the message list the frontend expects.
- The tool plugins in `tools/` are unchanged: `adk_app/tools.py` wraps each one as an ADK `BaseTool` whose declaration is the plugin's existing JSON schema, and runs `execute` with the per-turn context dict (`adk_app/turns.py`) on a single-worker thread — sequential tool semantics, and the embedding client's SQLite connection stays on one thread. The plugins' loop-control flags map onto ADK: `finish_turn` / `return_summary` / `record_memories` end an invocation via `skip_summarization`; `switch_mode` writes the new mode into session state, and the coach's instruction is a provider that recomputes the prompt from that state on every model call.
- The summarizer and memory agent are not ADK sub-agents (that would share the coach's session and expose `transfer_to_agent`); `adk_app/subagents.py` runs each spawn on its own `Runner` with a fresh session, which is the context isolation the brief asks for. `ask_library` is the ADK tool that awaits such a run.
- Iteration caps become `RunConfig(max_llm_calls=…)` (coach 20, summarizer 8, memory 3). An agent that ends a run without its terminal tool gets one reminder turn, hidden from the exported history.
- **Text replies.** The coach's plain text (outside thinking and tool calls) is delivered to the user as its reply through the same path as `send_message` (repeat guard, `has_responded`, queue), and a text-only response ends the turn — `send_message` / `finish_turn` still work but are optional. This suits models with proper reasoning separation; `BOTBASE_TEXT_REPLIES=0` restores the older rule where only `send_message` reaches the user and stray text is discarded (the frontend reads the mode from `/api/topics`).
- Tracing is an ADK plugin (`adk_app/tracing.py`) attached to every runner: model output, tool calls and results stream to the debug sub-pane of whichever agent is running, and persist to `trace_log`.
- Models come from `adk_app/models.py`: `LiteLlm` for OpenRouter and llama.cpp, ADK's native Gemini client for `gemini`. `SUMMARIZER_MODEL` / `MEMORY_MODEL` still pick a different model per helper.

**Coach (orchestrator, user-facing).** The existing topic-bot loop re-prompted with a multi-agent overlay. It owns the conversation and every user-facing action (`send_message`, `provide_file`, `open_course_page`, `switch_mode`, `finish_turn`) — writes stay single-threaded. Instead of searching the library itself, it delegates via two tools:

- **`ask_library(question)`** — spawns a fresh **summarizer** run and returns its structured summary.
- **`memory_search(query)`** — keyword search over past-session logs.

**Summarizer (read-only explorer).** A second `LlmAgent` with its own session per spawn and a read-only toolset (`search_resources`, `examine_resource`). It must finish by calling `return_summary`, which enforces a fixed output contract — `answer`, `key_points`, `resources` (ids validated against what the run actually retrieved), `confidence`, `notes` — stamped with `source` + `timestamp`. Retrieved resources are merged back into the coach's context so the cited ids resolve in `provide_file` / `open_course_page`. Effort is capped (8 tool calls) with scaling rules in the prompt. Set `SUMMARIZER_MODEL` in `.env` to run it on a cheaper model.

**User profiles.** The demo opens with a profile picker: select an existing profile or create a new one. All memory — injection, search, and writes — is scoped to the active profile (`user_id` travels with every chat request and is sticky per chat session server-side). No profile → no memory, so the single-agent baseline and simulator runs stay memory-free.

**Memory (template store + memory agent).** Privacy-first by construction: the store cannot hold free text. A memory is a template key from `memory_templates.py` plus validated slot values, where every slot is an enum from a fixed vocabulary (skills, assessments, stressor areas, event types, timeframes, …) or a reference to a library resource. Nothing the user literally said is ever persisted.

- *Write path* — after each completed coach turn, the **memory agent** (a third `LlmAgent`, run from `adk_app/subagents.py`) reads the turn's conversational surface plus the profile's existing notes and must answer through the `record_memories` tool, whose schema only admits registered templates (structured generation). Rejected entries bounce back with the reason so the model can retry; an empty list is a normal outcome. `MemoryStore.add_memories` re-validates and remains the single writer. Set `MEMORY_MODEL` in `.env` to run it on a cheaper model.
- *Templates* — grounded in the tool's three primary uses: **stress/coping** (`stress_reported`, `skill_introduced`, `skill_practiced`, `practice_commitment`, `assessment_result`), **task performance** (`upcoming_event`, `event_outcome`, `goal_set`), **distress** (`distress_supported` — records that distress was recognized and what support was given, never the details), plus continuity glue (`topic_discussed`, `resource_shared`, `preference_noted`). Some templates are flagged as follow-ups and surface first in the snapshot so the coach picks them up next session.
- *Read path* — the `memory_search` tool (pull, on the coach's initiative) over the profile's rendered notes.
- *Always-injected* — one hard-capped MEMORY SNAPSHOT block per turn: session count, open follow-ups, recent notes, all rendered deterministically from the stored templates.
- *Instrumentation* — every tool call and agent event is persisted to `trace_log` (the future push-channel spec and eval corpus, per the brief). Note: `trace_log`, `logs/`, and `logs.db` still capture raw traffic for debugging the demo — they are separate from the memory system and should be disabled for a real deployment.
- *Browser/editor* — `/memory` (also via the 🧠 Memory button in debug mode) is profile-aware: pick a profile, see its live injected snapshot and notes, and add/edit/delete notes through template-driven forms (every field is a dropdown from the registry — the browser can't store free text either). Useful for staging demo state and for showing an audience exactly what the bot can and cannot remember.
- *Scenarios* — the browser's Scenarios card snapshots one profile's notes under a name and loads them into any profile later (destructive replace, confirm-guarded), so a staged demo ("user returns the day after a session") is repeatable. On load, timestamps can be re-dated so the newest note is always N days ago, preserving relative spacing. Scenarios are plain JSON files under `scenarios/` holding template records only; sync or hand-edit them freely.

**Selecting the architecture.** Server default via `--arch multi|single` or `BOTBASE_ARCH`; the frontend's debug mode has a per-request selector, so you can run the brief's §6 comparison — same question, same budget, single vs. multi — from the UI.

**Debug sub-panes.** The Debug toggle now also reveals a bottom panel with one live sub-pane per agent — **Coach**, **Summarizer**, **Memory** — streaming each agent's model output, tool calls/results, spawns, structured summaries, and memory reads/writes as they happen.

Deliberately deferred (per the brief): proactive push channels, conflict detection / follow-up resolution on memory writes (notes are append-only for now), semantic/procedural memory tiers, a verification agent, and vector indexing of the notes.

## Customization

- **System prompt**: Edit `prompts/default.py` or create a new profile.
- **LLM backend**: `--backend` at startup; add a new recipe to `BACKENDS` / `make_model` in `adk_app/models.py` (anything LiteLLM supports is a one-liner).
- **Tools**: Drop new tool files into `tools/` and add them to a toolset.
