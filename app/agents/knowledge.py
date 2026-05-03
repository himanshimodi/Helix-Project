"""
KnowledgeAgent — answers product documentation questions via RAG.

Calls search_docs() to retrieve relevant chunks, then composes an answer
with mandatory chunk-ID citations. Created once at module load.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents.tools.search_docs import format_chunks_for_agent, search_docs
from app.settings import settings

KNOWLEDGE_INSTRUCTION = """You are the Helix product knowledge specialist.

When asked a question:
1. Call search_docs with the user's query to retrieve relevant documentation chunks.
2. Answer using ONLY the content from the returned chunks.
3. Every factual claim MUST cite its source chunk like this: [chunk_abc123ef].
4. If the retrieved chunks do not contain a clear answer, say:
   "I don't have documentation on that topic. Please contact support."
5. Do NOT answer from general knowledge — only from retrieved chunks.

Format citations inline, e.g.:
  "To rotate a deploy key [chunk_3a9f12bc], navigate to Settings → Deploy Keys…"
"""


async def _search_and_format(query: str, k: int = 5) -> str:
    """
    Search product docs and return formatted context with chunk IDs and scores.

    Args:
        query: The user's natural-language question.
        k: Number of chunks to retrieve (default 5).

    Returns:
        Formatted documentation chunks with chunk IDs for citation.
    """
    chunks = await search_docs(query=query, k=k)
    return format_chunks_for_agent(chunks)


knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    description=(
        "Answers questions about Helix product features, configuration, and "
        "how-to guides by searching the documentation. Use for: how-to questions, "
        "feature explanations, configuration help, error troubleshooting."
    ),
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[_search_and_format],
)
