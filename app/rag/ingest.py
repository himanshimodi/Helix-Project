"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 800 --chunk-overlap 100

Reads markdown files, chunks them, embeds, and writes to the vector store.

Chunking strategy: heading-aware (split on ## / ### headings).
Each section becomes one or more chunks. Long sections are sub-chunked by
sentence. This preserves natural document boundaries and keeps each chunk
semantically coherent — critical for retrieval quality on Markdown product docs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
import yaml


def extract_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (metadata_dict, body_without_frontmatter)."""
    match = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        metadata: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}
    body = text[match.end():]
    return metadata, body


def extract_metadata(file_path: Path, text: str) -> dict[str, Any]:
    """Extract YAML frontmatter metadata from a markdown file."""
    meta, _ = extract_frontmatter(text)
    return {
        "source": file_path.name,
        "title": str(meta.get("title", file_path.stem)),
        "product_area": str(meta.get("product_area", "general")),
    }


def _chunk_sentences(text: str, max_chars: int = 800, overlap_sentences: int = 1) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def chunk_markdown(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Heading-aware chunking: split on ## / ### boundaries, sub-chunk long sections.

    Preserves section context by keeping headings attached to their content.
    Each section ≤ chunk_size is one chunk; longer sections are sentence-split
    with `overlap` controlling the number of overlap sentences between sub-chunks.
    """
    _, body = extract_frontmatter(text)
    # overlap sentences kept between consecutive sub-chunks (min 1)
    overlap_sentences = max(1, overlap // 80)
    # Split on h2/h3 headings, keeping the heading with its section
    sections = re.split(r"\n(?=#{1,3} )", body)
    chunks: list[str] = []
    for section in sections:
        stripped = section.strip()
        if not stripped:
            continue
        if len(stripped) <= chunk_size:
            chunks.append(stripped)
        else:
            chunks.extend(
                _chunk_sentences(stripped, max_chars=chunk_size,
                                 overlap_sentences=overlap_sentences)
            )
    return [c for c in chunks if c.strip()]


def make_chunk_id(file_path: str, chunk_index: int) -> str:
    """Deterministic chunk ID based on file path and index."""
    raw = f"{file_path}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_chroma_client(persist_dir: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir)


def _embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    """Embed a batch of texts using Google text-embedding-004."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="retrieval_document",
    )
    embeddings = result.get("embedding", [])
    if embeddings and not isinstance(embeddings[0], list):
        embeddings = [embeddings]
    return embeddings  # type: ignore[return-value]


async def ingest_directory(
    docs_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    persist_dir: str = "./chroma_db",
    api_key: str = "",
    batch_size: int = 20,
) -> None:
    """Walk docs_path, chunk and embed every .md file, upsert into Chroma."""
    md_files = [f for f in docs_path.rglob("*.md")
                if not f.name.startswith(".")]
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    client = _get_chroma_client(persist_dir)
    collection = client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metas: list[dict[str, Any]] = []

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata = extract_metadata(file_path, text)
        chunks = chunk_markdown(text, chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(str(file_path), i)
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append({**metadata, "chunk_index": i})

    total = len(all_ids)
    print(f"Embedding {total} chunks in batches of {batch_size}…")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = all_texts[start:end]
        batch_ids = all_ids[start:end]
        batch_metas = all_metas[start:end]

        embeddings = await asyncio.to_thread(
            _embed_texts, batch_texts, api_key
        )
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        print(f"  Upserted chunks {start + 1}–{end}")

    print(f"Ingest complete. {total} chunks in vector store.")


def main() -> None:
    import os
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--persist-dir", type=str, default="./chroma_db")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not set — embeddings will fail.")

    asyncio.run(
        ingest_directory(
            args.path,
            args.chunk_size,
            args.chunk_overlap,
            persist_dir=args.persist_dir,
            api_key=api_key,
        )
    )


if __name__ == "__main__":
    main()
