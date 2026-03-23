"""RAG (Retrieval-Augmented Generation) module for document search and grading.

Provides vector similarity search against the document_embeddings table,
batch document grading via structured LLM output, query rewriting,
and token-budgeted result formatting.
"""

from __future__ import annotations

import logging
from typing import Any

import tiktoken
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy-initialized tiktoken encoder. Initialized on first use inside a worker
# thread to avoid blocking the async event loop (LangGraph blockbuster guard).
_TIKTOKEN_ENC: tiktoken.Encoding | None = None


def _get_tiktoken_enc() -> tiktoken.Encoding:
    """Return cached tiktoken encoder, initializing on first call."""
    global _TIKTOKEN_ENC  # noqa: PLW0603
    if _TIKTOKEN_ENC is None:
        try:
            _TIKTOKEN_ENC = tiktoken.get_encoding("o200k_base")
        except Exception:
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENC


class DocumentGrade(BaseModel):
    """Relevance grade for a single document."""

    index: int = Field(description="Document index (0-based)")
    is_relevant: bool = Field(description="Whether the document is relevant to the query")
    reasoning: str = Field(description="Brief reasoning for the relevance grade")


class GradeDocuments(BaseModel):
    """Batch relevance grades for retrieved documents."""

    grades: list[DocumentGrade] = Field(description="Relevance grades for each document")


async def generate_query_embedding(text: str, model: str) -> list[float]:
    """Generate an embedding vector for a search query using Upstage.

    Args:
        text: The query text to embed.
        model: Upstage embedding model name (e.g. 'solar-embedding-1-large').

    Returns:
        A list of floats representing the embedding vector.
    """
    from langchain_upstage import UpstageEmbeddings

    embeddings = UpstageEmbeddings(model=model)
    result = await embeddings.aembed_query(text)
    return result


async def search_documents(
    query_embedding: list[float],
    db_url: str,
    top_k: int,
    max_distance: float,
) -> list[dict[str, Any]]:
    """Search document_embeddings table using pgvector cosine distance.

    Args:
        query_embedding: The query embedding vector.
        db_url: PostgreSQL connection URL.
        top_k: Maximum number of results to return.
        max_distance: Maximum cosine distance threshold.

    Returns:
        A list of document dicts with content, metadata, and distance.
    """
    import psycopg
    from psycopg.rows import dict_row  # type: ignore[import-not-found]

    embedding_str = str(query_embedding)

    sql = (
        "SELECT de.id, de.job_id, de.element_index, de.page, de.element_type, "
        "de.content, de.metadata, de.chunk_index, "
        "f.filename, f.category, "
        "(de.embedding <=> %s::vector) AS distance "
        "FROM document_embeddings de "
        "JOIN parse_jobs pj ON de.job_id = pj.parser_job_id "
        "JOIN files f ON pj.file_id = f.id "
        "WHERE (de.embedding <=> %s::vector) < %s "
        "AND de.element_type IN ('text', 'table') "
        "ORDER BY distance ASC "
        "LIMIT %s"
    )

    async with await psycopg.AsyncConnection.connect(
        db_url, row_factory=dict_row
    ) as conn:
        cursor = await conn.execute(
            sql, (embedding_str, embedding_str, max_distance, top_k)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def search_raptor_summaries(
    query_embedding: list[float],
    db_url: str,
    top_k: int,
    max_distance: float,
) -> list[dict[str, Any]]:
    """Search raptor_summaries table for relevant cluster summaries.

    Uses pgvector cosine distance against pre-computed RAPTOR summaries.
    Each summary represents a cluster of related leaf chunks.

    Args:
        query_embedding: The query embedding vector.
        db_url: PostgreSQL connection URL.
        top_k: Maximum number of summary clusters to return.
        max_distance: Maximum cosine distance threshold.

    Returns:
        A list of summary dicts with job_id, raptor_level, cluster_id,
        content, metadata (containing source_indices), and distance.
    """
    import psycopg
    from psycopg.rows import dict_row  # type: ignore[import-not-found]

    embedding_str = str(query_embedding)

    sql = (
        "SELECT rs.job_id, rs.raptor_level, rs.cluster_id, rs.content, "
        "rs.metadata, (rs.embedding <=> %s::vector) AS distance "
        "FROM raptor_summaries rs "
        "WHERE (rs.embedding <=> %s::vector) < %s "
        "ORDER BY distance ASC "
        "LIMIT %s"
    )

    async with await psycopg.AsyncConnection.connect(
        db_url, row_factory=dict_row
    ) as conn:
        cursor = await conn.execute(
            sql, (embedding_str, embedding_str, max_distance, top_k)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def search_leaf_chunks_by_indices(
    job_id: str,
    element_indices: list[int],
    db_url: str,
) -> list[dict[str, Any]]:
    """Fetch specific leaf chunks by element indices from a RAPTOR cluster.

    Retrieves document chunks that belong to a matched RAPTOR cluster,
    identified by their element_index values from the cluster's
    source_indices metadata.

    Args:
        job_id: The parsing job identifier.
        element_indices: List of element_index values to retrieve.
        db_url: PostgreSQL connection URL.

    Returns:
        A list of document dicts compatible with grade_documents() and
        format_results(). Distance is set to 0.0 as a sentinel since
        these are cluster-matched, not distance-ranked.
    """
    import psycopg
    from psycopg.rows import dict_row  # type: ignore[import-not-found]

    sql = (
        "SELECT de.id, de.job_id, de.element_index, de.page, de.element_type, "
        "de.content, de.metadata, de.chunk_index, "
        "f.filename, f.category "
        "FROM document_embeddings de "
        "JOIN parse_jobs pj ON de.job_id = pj.parser_job_id "
        "JOIN files f ON pj.file_id = f.id "
        "WHERE de.job_id = %s "
        "AND de.element_index = ANY(%s) "
        "AND de.element_type IN ('text', 'table') "
        "ORDER BY de.element_index ASC"
    )

    async with await psycopg.AsyncConnection.connect(
        db_url, row_factory=dict_row
    ) as conn:
        cursor = await conn.execute(sql, (job_id, element_indices))
        rows = await cursor.fetchall()
        return [{**dict(row), "distance": 0.0} for row in rows]


async def grade_documents(
    query: str, documents: list[dict[str, Any]], model: str
) -> GradeDocuments:
    """Grade retrieved documents for relevance using batch structured output.

    Sends all documents in a single LLM call for O(1) grading.

    Args:
        query: The original user query.
        documents: List of document dicts from search_documents.
        model: LLM model name for grading.

    Returns:
        GradeDocuments with relevance grades for each document.
    """
    docs_text = "\n\n".join(
        f"[Document {i}]\n{doc.get('content', '')}"
        for i, doc in enumerate(documents)
    )

    prompt = (
        "You are a document relevance grader. Evaluate whether each document "
        "is relevant to the user's query. Be strict — only mark documents as "
        "relevant if they contain information that directly helps answer the query.\n\n"
        f"User query: {query}\n\n"
        f"Documents:\n{docs_text}\n\n"
        "Grade each document. Return a grade for every document index."
    )

    llm = ChatOpenAI(model=model, temperature=0, streaming=False)
    structured_llm = llm.with_structured_output(GradeDocuments)
    result = await structured_llm.ainvoke(prompt)

    # Validate indices are in range
    valid_grades = [
        grade for grade in result.grades if 0 <= grade.index < len(documents)
    ]
    result.grades = valid_grades

    return result


async def rewrite_query(original_query: str, model: str) -> str:
    """Rewrite a query to improve document retrieval results.

    Args:
        original_query: The original search query.
        model: LLM model name for rewriting.

    Returns:
        A rewritten query optimized for document retrieval.
    """
    llm = ChatOpenAI(model=model, temperature=0.3, streaming=False)
    prompt = (
        "Rewrite the following query to improve document retrieval results. "
        "Make it more specific and use alternative keywords that might match "
        "document content better. Return only the rewritten query.\n\n"
        f"Original query: {original_query}"
    )
    response = await llm.ainvoke(prompt)
    return str(response.content)


def format_results(documents: list[dict[str, Any]], max_tokens: int) -> str:
    """Format retrieved documents with source citations and token budget.

    Truncates at document boundaries. If the first document exceeds
    the budget, it is truncated to fit (guarantees at least one document).

    Args:
        documents: List of relevant document dicts.
        max_tokens: Maximum token budget for the formatted output.

    Returns:
        Formatted string with document content and source citations.
    """
    enc = _get_tiktoken_enc()

    parts: list[str] = []
    total_tokens = 0

    for i, doc in enumerate(documents):
        page_str = str(doc.get("page", "N/A"))
        distance = doc.get("distance", 0.0)
        filename = doc.get("filename", "unknown")
        content = doc.get("content", "")

        header = f"[문서 {i + 1}] 파일: {filename} | 페이지: {page_str} | 거리: {distance:.3f}"
        block = f"{header}\n{content}"
        block_tokens = len(enc.encode(block))

        if total_tokens + block_tokens > max_tokens:
            if i == 0:
                # First doc exceeds budget — truncate to fit
                available = max_tokens - len(enc.encode(header + "\n"))
                if available > 0:
                    tokens = enc.encode(content)[:available]
                    truncated_content = enc.decode(tokens)
                    parts.append(f"{header}\n{truncated_content}...")
                else:
                    parts.append(header)
            break

        parts.append(block)
        total_tokens += block_tokens

    return "\n\n".join(parts)
