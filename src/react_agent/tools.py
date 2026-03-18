"""Tools for web search and document retrieval.

Provides Tavily web search and RAG-based document retrieval
against embedded document content stored in PostgreSQL.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, cast

from langchain_tavily import TavilySearch

from react_agent.configuration import Configuration

logger = logging.getLogger(__name__)


async def search(query: str) -> dict[str, Any] | None:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


async def retrieve_documents(query: str) -> str:
    """Search through uploaded documents to find relevant information.

    Use this tool when the user asks about content from their uploaded
    documents, files, or knowledge base. This searches through embedded
    document content stored in the database using vector similarity.

    Args:
        query: The search query to find relevant document content.

    Returns:
        Formatted string of relevant document excerpts with source info,
        or a message indicating no relevant documents were found.
    """
    from react_agent.db import get_database_url
    from react_agent.rag import (
        format_results,
        generate_query_embedding,
        grade_documents,
        rewrite_query,
        search_documents,
    )

    config = Configuration.from_context()

    try:
        db_url = get_database_url()

        # Generate query embedding
        embedding = await generate_query_embedding(query, config.embedding_model)

        # Dimension mismatch guard
        if len(embedding) != config.embedding_dimensions:
            logger.error(
                "Embedding dimension mismatch: expected %d, got %d",
                config.embedding_dimensions,
                len(embedding),
            )
            return (
                "문서 검색 중 오류가 발생했습니다 (임베딩 차원 불일치). "
                "웹 검색을 시도해 보세요."
            )

        # Vector search
        results = await search_documents(
            embedding, db_url, config.rag_max_results, config.rag_max_distance
        )

        if not results:
            return "해당 문서에서 관련 내용을 찾지 못했습니다. 웹 검색을 시도해 보세요."

        # Batch grade documents
        grades = await grade_documents(query, results, config.rag_grading_model)
        relevant_indices = {g.index for g in grades.grades if g.is_relevant}
        relevant_docs = [
            doc for i, doc in enumerate(results) if i in relevant_indices
        ]

        # Rewrite and retry if no relevant docs
        attempts = 0
        while not relevant_docs and attempts < config.rag_max_rewrite_attempts:
            attempts += 1
            rewritten = await rewrite_query(query, config.rag_grading_model)
            embedding = await generate_query_embedding(
                rewritten, config.embedding_model
            )
            results = await search_documents(
                embedding, db_url, config.rag_max_results, config.rag_max_distance
            )
            if not results:
                break
            grades = await grade_documents(
                rewritten, results, config.rag_grading_model
            )
            relevant_indices = {g.index for g in grades.grades if g.is_relevant}
            relevant_docs = [
                doc for i, doc in enumerate(results) if i in relevant_indices
            ]

        if not relevant_docs:
            return "해당 문서에서 관련 내용을 찾지 못했습니다. 웹 검색을 시도해 보세요."

        return await asyncio.to_thread(
            format_results, relevant_docs, config.rag_max_response_tokens
        )

    except Exception:
        logger.exception("Error in retrieve_documents tool")
        return "문서 검색 중 오류가 발생했습니다. 웹 검색을 시도해 보세요."


TOOLS: List[Callable[..., Any]] = [search, retrieve_documents]
