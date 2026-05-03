"""
Test fixtures.

- client: async test client backed by in-memory SQLite
- mock_adk: patches pipeline.run at the ADK boundary so tests don't call the LLM
- seeded_session: creates a session via the API and returns its session_id
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.session import get_db
from app.main import app
from app.srop.pipeline import PipelineResult

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db: AsyncSession):
    """Async test client with DB overridden to in-memory SQLite."""
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch pipeline.run at the ADK boundary — tests never call the real LLM.

    Routing logic mirrors the real agent:
      - messages containing 'deploy', 'rotate', 'build', 'plan', 'how' → knowledge_agent
      - messages containing 'build list', 'recent builds', 'account', 'status' → account_agent
      - everything else → srop_root (smalltalk)
    """
    import uuid

    async def fake_run(
        session_id: str, user_message: str, db: AsyncSession
    ) -> PipelineResult:
        import re as _re
        msg = user_message.lower()

        def _has_word(word: str) -> bool:
            return bool(_re.search(r"\b" + _re.escape(word) + r"\b", msg))

        if any(_has_word(k) for k in ("deploy", "rotate", "what", "plan tier")) or (
            _has_word("how") and not _has_word("show")
        ):
            routed = "knowledge_agent"
            reply = (
                "To rotate a deploy key [chunk_abc12345678abcde], go to "
                "Settings → Deploy Keys and click Rotate. "
                "Your plan tier is pro [chunk_def12345678defgh]."
            )
        elif any(_has_word(k) for k in ("build", "builds", "account", "status", "recent")):
            routed = "account_agent"
            reply = "Your last 3 builds: bld_001 (failed), bld_002 (passed), bld_003 (failed)."
        else:
            routed = "srop_root"
            reply = "Hello! How can I help you today?"

        from app.db.models import AgentTrace, Message
        trace_id = str(uuid.uuid4())

        _kg = routed == "knowledge_agent"
        chunk_ids = (
            ["chunk_abc12345678abcde", "chunk_def12345678defgh"] if _kg else []
        )
        tool_calls = (
            [{"tool_name": "_search_and_format",
              "args": {"query": user_message},
              "result": "context"}]
            if _kg
            else []
        )

        from sqlalchemy import select, update

        from app.db.models import Session as DBSession
        from app.srop.state import SessionState

        result = await db.execute(
            select(DBSession).where(DBSession.session_id == session_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            from app.api.errors import SessionNotFoundError
            raise SessionNotFoundError(f"Session {session_id!r} not found.")

        state = SessionState.from_db_dict(row.state)
        state.last_agent = routed
        state.turn_count += 1

        db.add(AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed,
            tool_calls=tool_calls,
            retrieved_chunk_ids=chunk_ids,
            latency_ms=42,
        ))
        db.add(Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
            trace_id=trace_id,
        ))
        db.add(Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=reply,
            trace_id=trace_id,
        ))
        await db.execute(
            update(DBSession)
            .where(DBSession.session_id == session_id)
            .values(state=state.to_db_dict())
        )
        await db.commit()

        return PipelineResult(content=reply, routed_to=routed, trace_id=trace_id)

    monkeypatch.setattr("app.srop.pipeline.run", fake_run)
    monkeypatch.setattr("app.api.routes_chat.pipeline.run", fake_run)
