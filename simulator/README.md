# Bot Simulator

A standalone sandbox for testing a BotBase chatbot with simulated users.
It drives an OpenRouter-powered persona against a running BotBase server
(over its HTTP API only — no code dependency), records the conversation,
auto-evaluates it against a definable rubric with an LLM judge, and stores
everything (personas, runs, transcripts, evaluations, human reviews) in a
local SQLite database behind a small review web UI.

## Setup

```bash
pip install -r requirements.txt
```

Provide an OpenRouter key in `simulator/.env` (or it falls back to the
parent project's `.env`):

```
OPENROUTER_API_KEY=your_key_here
```

Optional environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SIM_PASSWORD` | `SimReview123` | Shared password for the review UI |
| `SIM_BOT_URL` | `http://localhost:5551` | Default BotBase server URL offered in the UI |
| `SIM_BOT_PASSWORD` | `AMIRADemo123` | Password used to log in to the BotBase server |

## Run

Start the bot under test (in the parent project) with whichever backend
you want to evaluate:

```bash
python server.py --backend llama_cpp   # or: --backend openrouter
```

Then start the simulator:

```bash
python server.py --port 5561
```

Open `http://localhost:5561`, enter the shared password, and set your
reviewer name (top right) so runs and reviews are attributed to you.

## Workflow

1. **Personas** — create simulated users: a freeform character profile
   (who they are, situation, goals, typing style), plus the OpenRouter
   model and temperature that plays them.
2. **Rubrics** — create named criteria sets (key, title, description,
   score range, weight) and pick the OpenRouter judge model. The judge
   scores each criterion with a justification; a weighted 0–100 score is
   computed from the results.
3. **Runs** — pick a persona, rubric, bot URL/profile, agent
   architecture, and max turns, then start. The transcript streams live;
   when the conversation ends (the persona decides it's done, or the turn
   cap is hit) the rubric evaluation runs automatically. The
   **architecture** selector (`multi` = coach + summarizer + memory,
   `single` = the one-agent baseline, blank = whatever the bot server
   defaults to) is passed per-run to `/api/chat-profile`, so you can A/B
   the two architectures against the same persona and rubric without
   restarting anything — this is exactly the single-vs-multi comparison
   the design brief calls for before expanding the system. The ↻ button
   next to the profile field probes the bot server for its profiles, its
   supported architectures, and its router (entry-point) profile. The
   profile field defaults to `router` — the profile a real user lands
   on — so simulated conversations exercise the actual coach persona;
   BotBase's bare `default` profile is a generic assistant that won't
   reflect real behaviour. Since BotBase picks its LLM *backend*
   at launch, use the free-text "backend note" field to record which
   backend the target server was running (run two instances on different
   ports to compare backends side by side).
4. **Review** — anyone with the password can open a run, read the
   transcript beside its rubric scores, re-evaluate with another rubric,
   and leave attributed star-ratings/comments.

## Files

```
server.py            Flask app — API + review UI
runner.py            Simulation loop + LLM judge
bot_client.py        HTTP client for the BotBase API
openrouter_client.py Minimal OpenRouter chat client
db.py                SQLite storage (simulator.db)
static/              Web UI
```

Persona and rubric definitions are snapshotted into each run, so editing
or archiving them later never changes historical results.

Bot turns use BotBase's async flow (`chat-profile` with `async: true`,
then polling): no HTTP request outlives a poll interval, so runs work
against a bot behind a reverse proxy with short read timeouts. Older
BotBase servers without async support are detected and handled
synchronously.

## Hosting at a non-root path

The web UI computes its API base from wherever the page is mounted
(`window.location.origin + pathname`), so it works unchanged at the
server root **and** behind a path-prefixed reverse proxy — no hardcoded
prefix. Point your proxy at the simulator and strip the prefix; e.g.
nginx:

```nginx
location = /simulator { return 301 /simulator/; }
location /simulator/ {
    proxy_pass http://127.0.0.1:5561/;   # trailing slash strips /simulator
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

The trailing slash on `proxy_pass` makes nginx replace `/simulator/` with
`/` before forwarding, which is what the Flask app expects; the `= /simulator`
redirect catches the bare no-slash URL. The server issues no redirects and
generates no absolute URLs of its own, so nothing else is needed. (The main
BotBase server uses the same pattern and can share the same proxy config
style.)
