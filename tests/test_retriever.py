"""
Unit tests for RAG components.
search_docs test requires the vector store to be seeded (run ingest.py first).
"""
from __future__ import annotations

import pytest


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings and must handle frontmatter."""
    from app.rag.ingest import chunk_markdown

    text = (
        "---\ntitle: Deploy Keys\nproduct_area: security\n---\n"
        "# Deploy Keys\n\n"
        "Deploy keys let you authenticate CI runners.\n\n"
        "## Rotating a Deploy Key\n\n"
        "To rotate, navigate to Settings → Deploy Keys and click Rotate. "
        "The old key is invalidated immediately.\n\n"
        "## Creating a Deploy Key\n\n"
        "Go to Settings → Deploy Keys → New Key. Give it a name and save."
    )
    chunks = chunk_markdown(text, chunk_size=200, overlap=40)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks), "Empty chunk found"


def test_chunk_ids_are_deterministic():
    """Same file + index always produces the same chunk ID."""
    from app.rag.ingest import make_chunk_id

    id1 = make_chunk_id("docs/deploy-keys.md", 0)
    id2 = make_chunk_id("docs/deploy-keys.md", 0)
    id3 = make_chunk_id("docs/deploy-keys.md", 1)

    assert id1 == id2, "Chunk IDs must be deterministic"
    assert id1 != id3, "Different indices must produce different IDs"
    assert id1.startswith("chunk_"), "IDs must start with 'chunk_'"
    assert len(id1) == len("chunk_") + 16


def test_frontmatter_extraction():
    """extract_frontmatter strips YAML block and returns metadata."""
    from app.rag.ingest import extract_frontmatter

    text = "---\ntitle: Builds\nproduct_area: ci-cd\n---\nBody text here."
    meta, body = extract_frontmatter(text)
    assert meta.get("title") == "Builds"
    assert meta.get("product_area") == "ci-cd"
    assert "Body text here." in body
    assert "---" not in body


def test_extract_metadata_uses_frontmatter():
    from pathlib import Path

    from app.rag.ingest import extract_metadata

    text = "---\ntitle: Webhooks\nproduct_area: integrations\n---\nContent."
    meta = extract_metadata(Path("docs/webhooks.md"), text)
    assert meta["title"] == "Webhooks"
    assert meta["product_area"] == "integrations"
    assert meta["source"] == "webhooks.md"


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """
    search_docs must return chunk IDs and scores in [0, 1].
    Requires: GOOGLE_API_KEY set and `python -m app.rag.ingest --path docs/` run first.
    Skipped automatically when vector store is empty or API key is missing.
    """
    import os
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set — skipping live embedding test")

    try:
        from app.agents.tools.search_docs import _get_collection
        col = _get_collection()
        if col.count() == 0:
            pytest.skip("Vector store is empty — run ingest.py first")
    except Exception:
        pytest.skip("ChromaDB not reachable")

    from app.agents.tools.search_docs import search_docs

    results = await search_docs("how to rotate a deploy key", k=3)
    assert len(results) > 0, "Expected at least one result"
    for r in results:
        assert r.chunk_id, "chunk_id must be non-empty"
        assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0,1]"
        assert r.content, "content must be non-empty"
