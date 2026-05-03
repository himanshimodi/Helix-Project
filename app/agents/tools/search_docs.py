"""
search_docs tool — used by KnowledgeAgent.

Queries the ChromaDB vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import chromadb


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )


def _embed_query(query: str) -> list[float]:
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query",
    )
    embedding: list[float] = result["embedding"]
    return embedding


SCORE_THRESHOLD = 0.3


async def search_docs(
    query: str,
    k: int = 5,
    product_area: str | None = None,
) -> list[DocChunk]:
    """
    Search the Helix product documentation for chunks relevant to the query.

    Args:
        query: Natural language question or search phrase.
        k: Maximum number of chunks to return (default 5).
        product_area: Optional metadata filter (e.g. 'security', 'ci-cd').

    Returns:
        List of DocChunk objects ordered by descending similarity score,
        each with a chunk_id for citation and a score in [0, 1].
    """
    import asyncio

    collection = _get_collection()
    query_embedding = await asyncio.to_thread(_embed_query, query)

    where: dict[str, Any] | None = (
        {"product_area": product_area} if product_area else None
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[DocChunk] = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for chunk_id, distance, doc, meta in zip(ids, distances, documents, metadatas):
        score = round(max(0.0, 1.0 - float(distance)), 4)
        if score < SCORE_THRESHOLD:
            continue
        chunks.append(
            DocChunk(
                chunk_id=str(chunk_id),
                score=score,
                content=str(doc),
                metadata=dict(meta) if meta else {},
            )
        )

    return sorted(chunks, key=lambda c: c.score, reverse=True)


def format_chunks_for_agent(chunks: list[DocChunk]) -> str:
    """Format retrieved chunks as readable context for the KnowledgeAgent."""
    if not chunks:
        return "No relevant documentation found."
    parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        parts.append(
            f"[{chunk.chunk_id}] (score: {chunk.score:.2f}, source: {source})\n"
            f"{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)
