# Helix SROP — Stateful RAG Orchestration Pipeline

AI Support Concierge handling knowledge questions (RAG over product docs) and account lookups in a single ongoing conversation, with state that survives process restarts.

---

## Setup (< 5 minutes)

**Prerequisites:** Python 3.11+, a Google API key with Generative AI access.

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY=your-key-here

# 3. Ingest docs into vector store
python -m app.rag.ingest --path docs/
# → Found N markdown files … Ingest complete. X chunks in vector store.

# 4. Run
uvicorn app.main:app --reload
# → http://localhost:8000/healthz {"status":"ok"}

# 5. Tests (LLM is mocked — no API key needed)
pytest -q
```

---

## Quick API Tour

```bash
# Create session
curl -s -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","plan_tier":"pro"}' | jq

# Knowledge question
SESSION_ID=<session_id>
curl -s -X POST http://localhost:8000/v1/chat/$SESSION_ID \
  -H "Content-Type: application/json" \
  -d '{"content":"How do I rotate a deploy key?"}' | jq

# Account query (same session — context persists)
curl -s -X POST http://localhost:8000/v1/chat/$SESSION_ID \
  -H "Content-Type: application/json" \
  -d '{"content":"Show my last 3 builds"}' | jq

# Trace
curl -s http://localhost:8000/v1/traces/<trace_id> | jq
```

---

## Architecture

```
POST /v1/chat/{session_id}
         │
         ▼
┌────────────────────────────────┐
│  pipeline.run()                │
│  1. Load SessionState (SQLite) │
│  2. Seed ADK session.state     │
│  3. Run ADK root_agent         │
│  4. Collect events → trace     │
│  5. Persist state + trace (DB) │
└──────────────┬─────────────────┘
               │  AgentTool routing (LLM decides)
        ┌──────┴───────┐
        ▼              ▼
 knowledge_agent   account_agent
 (_search_and_     (get_recent_builds,
  _format → RAG)    get_account_status)
        │
   ChromaDB          SQLite
   (doc chunks)      (sessions, messages,
                      agent_traces)
```

---

## Design Decisions

### State Persistence — Pattern 3 (SessionState injection via ADK session.state)

Store only `SessionState` (user_id, plan_tier, last_agent, turn_count) in the SQLite `sessions.state` JSON column. On each turn: load state → create a fresh ADK session seeded with it → run agent → save updated state back.

The root agent's instruction uses `{user_id}`, `{plan_tier}` etc. which ADK resolves from `session.state` at runtime via its built-in `inject_session_state` template mechanism.

**Why:** Survives restarts by design — the DB is the source of truth, ADK sessions are ephemeral. Minimal state footprint (no growing context window from full message history). Simple to reason about.

**Tradeoff vs Pattern 2:** The agent can't reference verbatim prior messages, only state variables. For richer multi-turn coherence, Pattern 2 (full message re-hydration) would be better at the cost of a growing context window each turn.

### Chunking — Heading-aware (Strategy C from rag-guide.md)

Split markdown on `##`/`###` headings, sub-chunk long sections by sentence with overlap. Chosen because product docs have clear section structure — keeping heading + content together improves retrieval precision. Fixed-size chunking would break across headings and lose semantic context.

### Embedding — Google `text-embedding-004`

Consistent with the ADK/Gemini stack already in the project. Uses `retrieval_document` task type at ingest and `retrieval_query` at query time (Google's recommended split for quality).

### Vector Store — ChromaDB (PersistentClient, cosine similarity)

No extra infrastructure. Stable SHA-256 chunk IDs prevent duplication on re-ingest. Use `--workers 1` with uvicorn (Chroma's `PersistentClient` is not safe for multi-worker setups).

---

## Known Limitations

- **Mock account data** — `get_recent_builds`/`get_account_status` return hardcoded data. Production would query a real DB or internal service API.
- **No auth** — All endpoints are unauthenticated. Production needs JWT or API-key verification on every route.
- **No rate limiting** — Add `slowapi` or a reverse proxy layer for production.
- **ADK runner per turn** — A new `InMemoryRunner` is created each chat turn (~1-5 ms overhead). A singleton runner with Pattern 2 session re-hydration would be faster at high volume.
- **Chroma single-worker** — `PersistentClient` requires `uvicorn --workers 1`. Swap for a hosted vector DB (Pinecone, Weaviate) for multi-worker deployments.

## What I'd Do With More Time

- Add streaming SSE (E3) — ADK's async event generator already streams; just wire `EventSourceResponse`
- Add idempotency keys (E1) — hash `(session_id, message)` and cache trace_id in a DB table
- Add LLM-as-judge reranker (E4) on top-k retrieval
- Persist full message history per turn (Pattern 2) for richer multi-turn coherence
- Add a real embedding cache to avoid re-embedding identical queries

## Time Spent

| Phase | Time |
|-------|------|
| Env setup + ADK API exploration | 45 min |
| DB schema + FastAPI boilerplate | 20 min |
| ingest.py — chunking + embedding + Chroma | 25 min |
| search_docs tool + account tools | 15 min |
| ADK agents (knowledge, account, orchestrator) | 30 min |
| pipeline.py — state + trace + ADK event parsing | 40 min |
| Routes + error handlers | 15 min |
| Tests + conftest mock_adk fixture | 25 min |
| README | 15 min |
| **Total** | **~3h 30min** |

## Extensions Completed

- [ ] E1: Idempotency
- [ ] E2: Escalation agent
- [ ] E3: Streaming SSE
- [ ] E4: Reranking
- [ ] E5: Guardrails
- [ ] E6: Docker
- [ ] E7: Eval harness
