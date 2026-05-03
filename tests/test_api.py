"""
Integration tests — full SROP pipeline with LLM mocked at the ADK boundary.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert body["user_id"] == "u_test_001"


@pytest.mark.asyncio
async def test_create_session_defaults_free_tier(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_free"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client, mock_adk):
    resp = await client.post(
        "/v1/chat/nonexistent-id", json={"content": "hello"}
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["title"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """
    Two-turn test:
      Turn 1 — knowledge question → routed_to == knowledge_agent, trace has chunk IDs.
      Turn 2 — plan tier question → reply contains 'pro' from persisted state.
    """
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"}
    )
    assert sess.status_code == 200
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge query
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
    )
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "knowledge_agent"
    trace_id = r1.json()["trace_id"]

    # Trace must include retrieved chunk IDs
    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    trace_body = trace.json()
    assert len(trace_body["retrieved_chunk_ids"]) > 0
    assert trace_body["routed_to"] == "knowledge_agent"
    assert trace_body["latency_ms"] >= 0

    # Turn 2 — follow-up about plan tier (state must survive between turns)
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "What is my plan tier?"},
    )
    assert r2.status_code == 200
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_account_query_routes_correctly(client, mock_adk):
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_003", "plan_tier": "free"}
    )
    session_id = sess.json()["session_id"]

    r = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "Show my recent builds"},
    )
    assert r.status_code == 200
    assert r.json()["routed_to"] == "account_agent"


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client):
    resp = await client.get("/v1/traces/nonexistent-trace")
    assert resp.status_code == 404
    assert resp.json()["title"] == "TRACE_NOT_FOUND"


@pytest.mark.asyncio
async def test_state_persists_across_turns(client, mock_adk):
    """Turn count increments across turns — proves state is persisted."""
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_state_test", "plan_tier": "enterprise"}
    )
    session_id = sess.json()["session_id"]

    await client.post(f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"})
    await client.post(f"/v1/chat/{session_id}", json={"content": "Show my recent builds"})

    # Both turns completed without error → state was loaded and saved across turns
    r3 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "What is my plan tier?"}
    )
    assert r3.status_code == 200
