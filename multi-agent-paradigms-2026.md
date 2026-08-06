# Modern Multi-Agent Paradigms — A Working Brief (August 2026)

*Prepared to inform the design of a summarizer / coach / memory chatbot.*

---

## 1. The one-paragraph summary

The field has converged on a small set of coordination topologies, and simultaneously converged on the conclusion that **most systems using them shouldn't be**. Harnesses (Claude Code, Hermes, LangGraph, OpenAI Agents SDK) now ship multi-agent primitives as standard, but both vendor guidance and 2026 academic results say the same thing: multi-agent architecture buys you exactly two things — **context isolation** and **parallelism** — and you pay for them in tokens, latency, and coordination failures. Split work along *context boundaries*, not along *job titles*.

For a prototype the operative advice is Anthropic's: start with the simplest thing that works and add complexity only when evidence supports it. **Jump to §6** for the build path; §2–§5 are the survey behind it and §7 is what to do when you expand.

---

## 2. The paradigm map

Five topologies dominate. They differ in *who holds the plan* and *where intermediate results live*.

| Pattern | Who decides next step | State lives in | Best for |
|---|---|---|---|
| **Orchestrator–worker (supervisor)** | Lead agent, turn by turn | Lead's context window | Cross-domain routing; the 2026 production default |
| **Peer team** | Lead assigns, peers self-claim | Shared task list / mailbox | Independent parallel workstreams that need to talk |
| **Workflow-as-code** | A script | Script variables | Repeatable fan-out over many items |
| **Handoff / swarm** | Whichever agent holds the baton | Passed conversation history | Persona switching in dialogue (support, sales) |
| **Debate / verification** | Fixed protocol + judge | Transcript | High-stakes correctness checks |

**Orchestrator–worker** is the safe starting point and what Anthropic recommends for teams new to multi-agent. **Handoff** (OpenAI Agents SDK, ex-Swarm) differs meaningfully: control transfers *fully* to the specialist, so the user effectively talks to a different agent. **Supervisor** keeps one agent facing the user and delegating behind it. That choice is a product decision, not a technical one — and for a coaching chatbot it matters a lot (see §6).

---

## 3. What the harnesses actually implement

### Claude Code
Four distinct primitives, deliberately separated:

- **Subagents** — spawned via a Task tool. Each gets a fresh context window, its own tool allowlist, runs to completion, returns a result. Subagents never talk to each other. Three properties do the work: isolation, focus, composability.
- **Agent Teams** (experimental, env-gated) — multiple full Claude Code sessions; one is team lead. Teammates share a task list with dependency tracking and auto-unblocking, plus an inbox-based messaging system for direct peer communication. Context is *not* shared — only explicit messages cross the boundary.
- **Skills** — instructions Claude follows, progressively disclosed rather than preloaded.
- **Dynamic Workflows** — Claude writes a JavaScript orchestration script; a runtime executes it out-of-process, so intermediate results live in script variables and only the final answer lands in Claude's context. Caps at 16 concurrent agents / 1,000 per run. The bundled `/deep-research` workflow fans out searches, has agents cross-check each other's claims, and filters out claims that fail verification.

The key architectural insight in the workflow primitive: **moving the plan out of the context window is itself a context-engineering technique**, and it enables adversarial review as a repeatable pattern rather than an ad-hoc prompt.

### Hermes (Nous Research)
The most complete open-source harness right now, and the more relevant reference for your build because it's session- and memory-centric rather than repo-centric. Notable choices:

- **Three-tier system prompt assembly**: *stable* (identity, tool guidance, skills index), *context* (project files read from cwd, with prompt-injection scanning before load), *volatile* (memory snapshots, user profile, timestamp). Tiering keeps prompt prefixes cache-friendly and makes invariants easy to reason about.
- **Lineage-based compression**: when context compresses, it closes the current session row, creates a *child* session seeded by the summary, rotates the session ID, and records parent–child lineage. You get a chain, not a repeatedly-rewritten transcript.
- **Sessions as infrastructure**: SQLite + FTS5, with a model-facing `session_search` tool so recall is a decision the model makes at runtime rather than static injection.
- **Tool registration separated from tool exposure**: a broad installed library, a narrow model-visible surface per run.
- **Delegation without durable child lifecycle**: `delegate_task` gives children their own task ID and returns a structured summary, but children die with the parent. Reviewers flag this as the main gap.

### Everything else, briefly
LangGraph (explicit graph, checkpointing, time-travel) for when *you* want to own the control flow; OpenAI Agents SDK (handoffs, guardrails, tracing) for model-owned routing; CrewAI for role-based config; AutoGen's GroupChat is the canonical "collaboration" pattern and is the least durable in production. MCP (tool access) + A2A (agent coordination) are settling as the two-layer interop default, both now under the Linux Foundation's Agentic AI Foundation.

---

## 4. What the research says (the uncomfortable part)

Three findings you should design against:

**MAST — the failure taxonomy** (Cemri et al., NeurIPS 2025; 1,600+ annotated traces across 7 frameworks, κ = 0.88). 14 failure modes in 3 buckets: specification/system-design issues (~42%), inter-agent misalignment (~37%), task verification gaps (~21%). The headline: most failures are *architecture* failures, not model failures. Ambiguous role definitions, poor decomposition, duplicated roles, and missing termination conditions dominate.

**Compute-normalized comparison** (Tran & Kiela, Stanford, arXiv:2604.02460). When you hold the *thinking-token budget* constant, single agents match or beat multi-agent systems on multi-hop reasoning across three model families. The theoretical backbone is the Data Processing Inequality: every handoff can lose information, never add it. Multi-agent becomes competitive precisely when a single agent's context utilization degrades — which reframes the debate from "which is better" to "where is the boundary." Most published multi-agent wins simply spent more compute.

**Anthropic's own variance analysis**: in their research system, token usage alone explained ~80% of performance variance; tool-call count and model choice explained most of the rest. Prompt phrasing was not a primary driver. The practical read: if a single agent plateaus, first ask whether it's *context-bound* — not whether the prompt needs polish.

Against this, the pro-multi-agent evidence is real but narrow: Anthropic's research system beat single-agent Opus 4 by ~90% on their internal eval, at roughly 15× the tokens of a chat interaction, on a task class (breadth-first search across independent threads) that decomposes cleanly. Their more recent guidance puts typical multi-agent overhead at 3–10× for equivalent tasks.

**Cognition's counter-position** ("Don't Build Multi-Agents") has aged into a useful principle rather than a prohibition: keep *writes* single-threaded; extra agents are safest when they contribute intelligence (reading, analyzing) rather than actions.

---

## 5. Memory: state of the art

Your memory agent is the component where the literature is richest and least settled.

**The taxonomy everyone now uses** — sensory, working, episodic, semantic, procedural. The critical operational insight is that **each type has a different write path and a different retrieval logic**, and most production systems collapse them into one retrieval problem:

- *Working* — the live context window. Manage as a **budget** problem (compaction, prioritization), not a retrieval problem. Treating it as retrieval is a category error.
- *Episodic* — what happened, when. Written automatically by logging. Semantic-similarity search over episodic logs is the most common mistake; temporal and sequential queries need different machinery.
- *Semantic* — durable facts, preferences, profile. Written by an **extraction step**, ideally with conflict detection: compare new facts against existing entries and merge / update / flag rather than blindly append.
- *Procedural* — reusable strategies and skills. Should be written by a **deliberate promotion step with validation** — the one write path that should never be fully automatic. Tooling here is still early-stage across the ecosystem.

**Open problems, per the field's own assessment**: staleness and forgetting are unsolved at the tooling layer; systems treat change as replacement rather than evolution across sessions. GateMem (arXiv:2606.18829) benchmarks shared-memory governance and finds *no* method simultaneously achieves good utility, robust access control, and reliable forgetting — retrieval and external-memory approaches still leak deleted or unauthorized information. "Governed Memory" (arXiv:2603.17787) catalogs five structural failures in multi-agent memory: silos, governance fragmentation, unstructured memories unusable downstream, redundant context delivery, and silent quality degradation with no feedback loop.

**Memory contagion** is worth knowing about: shared-memory architectures propagate poisoned or biased stored content more readily than independent-memory ones, with no established safe contamination threshold.

---

## 6. The prototype path

*Scope: first working version, built on the existing librarian agent (which already runs its own tool-based search). Base functionality first; §7 is the guidance for expanding later.*

### What the librarian already gives you

The librarian is an LLM in a loop with search tools. That is exactly the machinery both the summarizer and the coach need, and it means **the pull path is already built**. Concretely:

- **Coach** = librarian, re-prompted. Same loop, same tool-calling infrastructure; new system prompt, new output style, and its resource-presentation behaviour becomes the coaching surface rather than being thrown away.
- **Summarizer** = a second instance of the same agent with a different system prompt, a **read-only** tool set, and a different output contract — compressed units for another model to consume rather than resources for a human to read. Run it on a cheaper model.
- **Memory** = the only genuinely new build, and for v0 it need not be an agent at all (see below).

Forking one working agent into two roles is a far shorter path than standing up an orchestration framework, and it keeps you inside a codebase you already understand.

### Build pull first, push second

Your design has both helpers pushing proactively. For the prototype, **build only the pull path**, then log every request the coach makes.

This is not a retreat from the always-on design — it's the only way to build it well. A push channel needs a relevance model and a token budget, and you currently have no data to calibrate either. After a few dozen real sessions, your tool-call logs tell you exactly what the coach asked for, when, and what it asked for repeatedly. *That* is the specification for the push. Building the push first means guessing at it, and a miscalibrated push is worse than none: it burns the context budget it was meant to protect and it's hard to attribute failures to.

Sequence: v0 pull-only → read the logs → v1 adds push for the patterns that actually recur, with a silence action and a hard token cap.

### Minimum viable v0

| Component | v0 implementation |
|---|---|
| **Coach** | Librarian re-prompted. Holds the conversation, owns the loop, calls two tools. |
| **Summarizer** | Same agent, read-only tools, cheap model, compressed-summary output contract. Exposed to the coach as one tool: `ask_repo(question) → summary`. |
| **Memory (write)** | Plain logging after each turn. No LLM, no extraction yet. |
| **Memory (read)** | `memory_search(query) → hits` over the log. Keyword/FTS is fine; skip embeddings. |
| **Always-injected** | One small profile block, hard-capped (start at a few hundred tokens). Nothing else auto-enters context. |

Everything else waits.

### Cheap now, expensive later — do these anyway

Four things cost almost nothing at prototype stage and are painful to retrofit, because they're contracts rather than features:

1. **A fixed output contract for the summarizer.** Decide now what a "usable unit" is — a struct with fields, not free text. Anthropic's guidance is explicit that orchestrator verification is more reliable when subagents return structured data. Changing this later means rewriting both ends.
2. **Single-writer on memory state.** One owner of writes, versioned. This is IntelliCode's policy and it's what makes state updates auditable. Retrofitting it after two components write concurrently is a rewrite.
3. **Source + timestamp on everything returned.** Not a full provenance schema — two fields. It's what lets you debug a bad answer later, and what lets the coach attribute rather than assert.
4. **Log every tool call, in and out.** This is your push specification, your eval set, and your failure corpus. MAST ships as a pip-installable annotator (`agentdash`) if you later want to classify traces against the taxonomy.

### Explicitly deferred

Proactive push on both channels · silence/no-op calibration · conflict detection on memory writes · semantic and procedural memory tiers · a verification subagent · code-graph or vector indexing of the repo · compaction and lineage · memory governance and access control · agent teams or peer messaging · A2A.

None of these are wrong. They're all answers to problems you haven't hit yet, and several of them (indexing, verification) may turn out to be unnecessary if the librarian's existing agentic search performs well enough — which, per the 2026 benchmarks, it often does.

### The check to run before expanding

Before adding a fourth thing or a push channel, run the comparison the literature keeps asking for: **a single coach agent with the repo search tools bolted on directly, at the same token budget.** If it matches your three-component system, the architecture isn't earning its cost yet and the next increment should be prompt and contract work, not more agents. If it doesn't, you'll know precisely which component closed the gap.

---

## 7. Design notes for the expanded system

*Everything below is for the version after the prototype. §6 is what to build now.*


Three components: summarizer, coach, memory. A fourth — a "librarian" doing direct resource retrieval and presentation — was folded into the coach. **That merge was the right call**, and for a reason worth generalizing: the librarian was a *tool*, not an agent. Retrieve-and-present is a function call; it has no reasoning loop of its own. Splitting it out would have been a problem-centric split (dividing by type of work) rather than a context-centric one, and it would have added a handoff for nothing.

Note the structural consequence: with the librarian gone there is no separate orchestrator, so **the coach is the orchestrator**. That's the standard supervisor topology, one level deep — coach faces the user, holds the conversation, and delegates to summarizer and memory behind it. Make that explicit in the prompts rather than leaving it emergent; ambiguous role definition is the single largest MAST failure category.

**Then apply the librarian test to the other two.** An agent is an LLM in a loop that decides its own control flow; anything else is a step in a pipeline. If the summarizer iteratively searches (plan → grep → read → refine → repeat) it is genuinely an agent. If it takes a file and returns a summary in one pass, it's a function with an isolated context — which is fine and *cheaper*, but it means you don't need multi-agent machinery for it. Same question for memory: extraction and consolidation are often a background LLM call, not an agent. If both turn out to be functions, what you have is a single agent with two context-isolated helpers — a simpler and more defensible architecture than a three-agent system, and one that sidesteps most of §4.

### Is your split context-centric or problem-centric?

Anthropic's decomposition test is the sharpest tool here. Good boundaries: independent research paths, components with clean interfaces, and **blackbox verification** (a verifier needs no implementation context). Bad boundaries: sequential phases of the same work, tightly coupled components, anything needing frequent shared-state sync.

Scoring your three:

- **Summarizer → strong boundary.** It ingests a large, high-volume corpus (repo) and returns a compact artifact. This is textbook context isolation: >1000 tokens in, well-defined extraction criteria, most of it irrelevant to the coach's reasoning. Keep it separate. This is the component that most justifies your multi-agent design.
- **Memory → strong boundary, with caveats.** Extraction and consolidation are genuinely separable background work. But it must not become a *write* competitor to the coach.
- **Coach → the risk.** If the coach and summarizer are doing sequential phases of the same reasoning ("understand the repo" → "explain it to the user"), you're in telephone-game territory. Anthropic's reported experiment with planner/implementer/tester/reviewer roles found subagents spending more tokens coordinating than working. Watch for the coach re-asking the summarizer for things it already had.

### Specific recommendations

1. **Adopt the single-writer principle for memory.** IntelliCode (EACL 2026) is the closest published analogue to your system: six pedagogical agents over a *centralized, versioned learner state*, each acting as a pure transformation under a single-writer policy. That gives auditable state updates. Do this — one owner of writes, versioned, with a diff you can inspect.
2. **Split memory's write paths by type.** Episodic logging: automatic. Semantic facts about the user: extraction with conflict detection. Procedural ("how this user learns best," "approaches that failed"): promotion step, gated. Hermes ships exactly this shape with optional `memory.write_approval` / `skills.write_approval` gates.
3. **Give the coach a `memory_search`-style tool, not just injection.** Hermes's `session_search` pattern — recall as a runtime decision by the model — beats always-injecting a memory blob. Keep a small always-loaded profile snapshot (volatile tier) plus on-demand search. This also bounds your per-turn token cost.
4. **Prefer agentic search over pre-indexing for the summarizer — but measure it.** The 2026 pattern in coding agents is just-in-time context loading (lightweight identifiers → load on demand) rather than vector-indexing the repo. Keyword search via agentic tool use reaches most of RAG-level performance without a vector DB. That said, code-graph indexing results (Tree-sitter KG over MCP) report large reductions in token use and tool calls on multi-repo evaluations. If your repo is stable and large, an index may pay for itself; if it's small or churning, agentic search is less machinery.
5. **Use the read-only explorer pattern.** Run the summarizer on a smaller/cheaper model, read-only, in its own context, returning a compressed summary rather than raw file contents. This is how Claude Code's Explore subagent works and it's the cheapest reliable win available to you.
6. **Write explicit termination conditions and effort scaling.** Missing termination is a named MAST mode. Anthropic's research system encodes rules like: simple lookup = one agent, 3–10 tool calls; comparison = 2–4 subagents, 10–15 calls each. Put numbers in your prompts.
7. **Don't add a verification agent yet.** Verification subagents work well, but capable orchestrator models increasingly evaluate subagent output directly. Add one only if you find the coach passing through bad summaries — and if you do, give it concrete criteria, because the dominant failure is declaring success after a token check.
8. **Instrument before you optimize.** Trace every handoff. MAST ships as a pip-installable annotator (`agentdash`) — you can classify your own failure traces against the taxonomy rather than guessing.
9. **Benchmark against a single agent with a bigger budget.** This is the discipline the Tran & Kiela paper is really arguing for. Before attributing a quality gain to your architecture, confirm it isn't just spend. If a single coach agent with compaction + note-taking + a repo search tool matches your four-agent system at equal tokens, you have your answer.

### The push/pull question — the central design decision

Both your helpers are **dual-mode**: they push unsolicited content to the coach every turn *and* answer direct tool queries. Those two modes have completely different engineering profiles and should be built as different things.

**The pull path is well-supported and cheap.** Hermes's `session_search`, Letta's recall/archival tiers, just-in-time context loading — recall as a runtime decision by the model, firing only when needed. Build this as a tool. Nothing controversial here.

**The push path is where the design lives or dies**, and there is now direct evidence on it. *Remember When It Matters* (arXiv:2607.08716, Jul 2026) builds almost exactly your memory component: a memory agent running alongside an unmodified action agent, updating a structured bank at fixed intervals and then deciding whether to inject a concise reminder into the next call — **or to stay silent**. It reports +8.3 pp on Terminal-Bench 2.0 and +6.8 pp on τ²-Bench. Two ablations matter to you:

- *Full-bank context* — maintain the bank but expose all of it every step, with no relevance selection. This is passive context augmentation. It beats the baseline but **trails the full system**.
- *Always inject* — keep the synthesized reminder but remove the silence action. Tests whether the no-op is merely an efficiency trick. It isn't; the paper finds intervention calibration is an essential capability, not an optimization.

**So: "always on" should mean always *evaluating*, not always *speaking*.** Give both the summarizer and the memory an explicit no-op action and make choosing it a first-class, rewarded outcome rather than a failure to find something. An agent that must justify its existence every turn will find something to say every turn.

**Split the push by scope, not by component.** CALMem (arXiv:2605.20724) draws the line usefully: current-session facts are injected automatically every turn; cross-session facts are available only through an explicit `recall_facts` call. The reason is structural — it stops the injection block growing without bound as the store accumulates. Map that onto your memory agent directly: live conversational state auto-injects, prior-session material comes in only on demand or via a deliberately narrow, high-confidence reminder.

**Budget the push explicitly.** Anything always in context is your highest-cost tier — every token there is a token unavailable for retrieved information or conversation history. CALMem drives injection off a context fill ratio (current tokens / capacity, compacting at ~0.8); Hermes scales its summary budget to ~20% of compressed content with a 2k floor and 12k ceiling. Pick a mechanism, but pick one. Unbudgeted push produces the paradox where the system built to extend context is what exhausts it.

### The two-writer problem

Your summarizer and your memory both push into the same attention budget, independently, with no arbitration. Nobody deduplicates them; nobody resolves contradiction; nobody enforces a combined cap. Route both through **one assembly point** that applies a priority order and a single token ceiling — this is Hermes's volatile tier, and it's a context-assembly concern rather than an agent concern. It's also the cheapest place to add provenance.

### The verification gap

You've said the coach has no specialized knowledge of the task. That's a defensible design — generalist reasoner, specialist retrieval — but it has a consequence: **the coach cannot evaluate what it is handed.** It has no independent basis to detect a wrong summary or a bad memory extraction, so errors pass straight through to the user. Verification gaps are ~21% of MAST failures.

Two mitigations, in order of cost:

1. **Provenance on every pushed item** — source, timestamp, confidence, when last confirmed. Cheap, and it lets the coach say "according to the repo docs, as of X" rather than asserting. It also makes bad items expirable and debuggable.
2. **Conflict detection on the memory write path** — compare each new fact against existing entries and merge, update, or flag rather than append. This is Mem0's core architectural move. Without it, contradictory facts accumulate silently.

Note also that the summarizer is judging relevance to a conversation it only partially observes. Cognition names this failure *implicit state sharing*: agents assume they share state with their counterpart when they don't. Feed the summarizer the actual current turn, not a compressed proxy of it — and if the push happens speculatively *before* the user's message rather than after it, expect the relevance judgments to be noticeably worse.

---

## 8. Reading list

**Vendor / practitioner**
- Anthropic — *Building multi-agent systems: When and how to use them* (Jan 2026) — the decision framework and context-centric decomposition. Most directly useful document here.
- Anthropic — *Effective context engineering for AI agents* — compaction, structured note-taking, sub-agent architectures.
- Anthropic — *How we built our multi-agent research system* — orchestrator-worker, effort scaling, the 15× figure.
- Anthropic — *The new rules of context engineering for Claude 5 generation models* (Jul 2026) — 80%+ system prompt reduction; rules → judgment; progressive disclosure. Relevant if you're over-instructing.
- Claude Code docs: `/docs/en/sub-agents`, `/docs/en/agent-teams`, `/docs/en/workflows`.
- Arize — *How Hermes implements an open source agent harness architecture* (Jun 2026) — nine-component harness model.
- Cognition — *Don't Build Multi-Agents* — the single-writer argument.

**Academic**
- Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* — arXiv:2503.13657 (MAST).
- Tran & Kiela, *Single-Agent LLMs Outperform Multi-Agent Systems… Under Equal Thinking Token Budgets* — arXiv:2604.02460.
- David & Ghosh, *IntelliCode: A Multi-Agent LLM Tutoring System with Centralized Learner Modeling* — EACL 2026 / arXiv:2512.18669. Closest published system to yours.
- *Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents* — arXiv:2607.08716. **Read this one first** — it's your memory component, with ablations on exactly the push question.
- *CALMem: Application-Layer Dual Memory for Conversational AI* — arXiv:2605.20724. Automatic injection vs. explicit retrieval; token-budget-adaptive injection.
- *Rethinking Memory Mechanisms of Foundation Agents* — arXiv:2602.06052 (five-type memory taxonomy).
- *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers* — arXiv:2603.07670.
- Ren et al., *GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents* — arXiv:2606.18829.
- Taheri, *Governed Memory: A Production Architecture for Multi-Agent Workflows* — arXiv:2603.17787.

---

*Sources retrieved August 2026. Fast-moving area — the harness details in §3 in particular have a short half-life.*
