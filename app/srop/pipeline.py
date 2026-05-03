"""
SROP pipeline — one turn of the conversation.

Flow per turn:
  1. Load SessionState from DB  (404 if session missing)
  2. Create a fresh ADK session seeded with that state
  3. Run root_agent via InMemoryRunner — ADK resolves {var} placeholders from state
  4. Collect events → routing decision + tool calls + chunk IDs for the trace
  5. Write AgentTrace row to DB
  6. Persist updated SessionState back to DB
  7. Return PipelineResult

State-persistence strategy: Pattern 3 — store only SessionState in the DB,
inject it into the agent's instruction template at runtime via ADK session.state.
No full message history needed; the agent sees state context on every turn.
Survives process restarts because state is in SQLite, not in-process memory.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import structlog
from google.genai import types as genai_types
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger()

APP_NAME = "helix_srop"


def _make_runner():  # InMemoryRunner imported lazily to avoid circular imports
    """Build a one-turn InMemoryRunner with the root agent."""
    from google.adk.runners import InMemoryRunner

    from app.agents.orchestrator import root_agent
    return InMemoryRunner(agent=root_agent, app_name=APP_NAME)


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def _load_session_state(session_id: str, db: AsyncSession) -> SessionState:
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise SessionNotFoundError(f"Session {session_id!r} not found.")
    return SessionState.from_db_dict(row.state)


async def _save_session_state(
    session_id: str, state: SessionState, db: AsyncSession
) -> None:
    await db.execute(
        update(Session)
        .where(Session.session_id == session_id)
        .values(state=state.to_db_dict())
    )
    await db.flush()


async def _write_messages(
    session_id: str,
    trace_id: str,
    user_message: str,
    assistant_reply: str,
    db: AsyncSession,
) -> None:
    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
            trace_id=trace_id,
        )
    )
    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=assistant_reply,
            trace_id=trace_id,
        )
    )
    await db.flush()


async def _write_trace(
    trace_id: str,
    session_id: str,
    routed_to: str,
    tool_calls: list[dict[str, Any]],
    chunk_ids: list[str],
    latency_ms: int,
    db: AsyncSession,
) -> None:
    db.add(
        AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed_to,
            tool_calls=tool_calls,
            retrieved_chunk_ids=chunk_ids,
            latency_ms=latency_ms,
        )
    )
    await db.flush()


def _extract_chunk_ids(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Pull chunk IDs out of search_and_format tool results."""
    import re
    ids: list[str] = []
    for tc in tool_calls:
        result_str = str(tc.get("result", ""))
        ids.extend(re.findall(r"chunk_[0-9a-f]{16}", result_str))
    return list(dict.fromkeys(ids))  # deduplicate, preserve order


async def _run_adk(
    state: SessionState,
    user_message: str,
) -> tuple[str, str, list[dict[str, Any]]]:
    """
    Run one ADK turn. Returns (reply, routed_to, tool_calls).
    Raises UpstreamTimeoutError on LLM timeout.
    """
    runner = _make_runner()

    adk_session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=state.user_id,
        session_id=adk_session_id,
        state=state.to_adk_state(),
    )

    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )

    routed_to = "srop_root"
    tool_calls: list[dict[str, Any]] = []
    reply = ""
    pending_calls: dict[str, dict[str, Any]] = {}

    async def _collect() -> None:
        nonlocal routed_to, reply
        async for event in runner.run_async(
            user_id=state.user_id,
            session_id=adk_session_id,
            new_message=new_message,
        ):
            author: str = getattr(event, "author", "srop_root")

            for fc in event.get_function_calls():
                call_id = getattr(fc, "id", fc.name)
                entry: dict[str, Any] = {
                    "tool_name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                    "result": None,
                }
                pending_calls[call_id] = entry
                tool_calls.append(entry)

            for fr in event.get_function_responses():
                resp_id = getattr(fr, "id", fr.name)
                matched = pending_calls.get(resp_id)
                if matched is None:
                    for tc in tool_calls:
                        if tc["tool_name"] == fr.name and tc["result"] is None:
                            matched = tc
                            break
                if matched is not None:
                    matched["result"] = fr.response

            if event.is_final_response():
                if author not in ("srop_root", "user", ""):
                    routed_to = author
                if event.content and event.content.parts:
                    reply = "".join(
                        p.text for p in event.content.parts if p.text
                    )

    try:
        await asyncio.wait_for(_collect(), timeout=settings.llm_timeout_seconds)
    except TimeoutError as exc:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        ) from exc

    if not reply:
        reply = "I'm sorry, I couldn't generate a response. Please try again."

    return reply, routed_to, tool_calls


async def run(
    session_id: str,
    user_message: str,
    db: AsyncSession,
) -> PipelineResult:
    """Run one SROP pipeline turn. Called by the /chat route."""
    trace_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(
        session_id=session_id, trace_id=trace_id
    )

    state = await _load_session_state(session_id, db)
    log.info("pipeline_started", turn=state.turn_count, user_id=state.user_id)

    t0 = time.monotonic()
    reply, routed_to, tool_calls = await _run_adk(state, user_message)
    latency_ms = int((time.monotonic() - t0) * 1000)

    chunk_ids = _extract_chunk_ids(tool_calls)

    await _write_trace(
        trace_id, session_id, routed_to, tool_calls, chunk_ids, latency_ms, db
    )
    await _write_messages(session_id, trace_id, user_message, reply, db)

    state.last_agent = routed_to
    state.turn_count += 1
    await _save_session_state(session_id, state, db)

    await db.commit()

    log.info(
        "pipeline_done",
        routed_to=routed_to,
        latency_ms=latency_ms,
        chunks=len(chunk_ids),
    )
    return PipelineResult(content=reply, routed_to=routed_to, trace_id=trace_id)
