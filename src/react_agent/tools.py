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
    logger.info(
        "[tool:search] query=%r, max_results=%d",
        query,
        configuration.max_search_results,
    )
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    result = cast(dict[str, Any], await wrapped.ainvoke({"query": query}))
    result_count = len(result.get("results", [])) if isinstance(result, dict) else 0
    logger.info("[tool:search] done — results=%d", result_count)
    return result


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
        search_leaf_chunks_by_indices,
        search_raptor_summaries,
    )

    config = Configuration.from_context()
    logger.info("[tool:retrieve_documents] query=%r", query)

    try:
        db_url = get_database_url()

        # Generate query embedding
        embedding = await generate_query_embedding(query, config.embedding_model)
        logger.info(
            "[tool:retrieve_documents] embedding — model=%s, dims=%d",
            config.embedding_model,
            len(embedding),
        )

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

        # --- RAPTOR 2-stage retrieval ---
        results: list[dict[str, Any]] = []

        if config.enable_raptor:
            try:
                clusters = await search_raptor_summaries(
                    embedding,
                    db_url,
                    config.raptor_top_k,
                    config.raptor_max_distance,
                )
                logger.info("RAPTOR stage 1: found %d clusters", len(clusters))

                if clusters:
                    # Extract source_indices defensively
                    indices_by_job: dict[str, set[int]] = {}
                    for cluster in clusters:
                        metadata = cluster.get("metadata")
                        if isinstance(metadata, str):
                            import json

                            try:
                                metadata = json.loads(metadata)
                            except (json.JSONDecodeError, TypeError):
                                metadata = {}
                        if not isinstance(metadata, dict):
                            metadata = {}
                        source_indices = metadata.get("source_indices", [])
                        valid_indices = [
                            idx for idx in source_indices if isinstance(idx, int)
                        ]
                        if valid_indices:
                            job_id = cluster.get("job_id", "")
                            if job_id:
                                indices_by_job.setdefault(job_id, set()).update(
                                    valid_indices
                                )

                    # Fetch leaf chunks per job_id
                    if indices_by_job:
                        all_leaf_chunks: list[dict[str, Any]] = []
                        for job_id, indices in indices_by_job.items():
                            chunks = await search_leaf_chunks_by_indices(
                                job_id, sorted(indices), db_url
                            )
                            all_leaf_chunks.extend(chunks)
                        if all_leaf_chunks:
                            results = all_leaf_chunks
                            logger.info(
                                "RAPTOR stage 2: fetched %d leaf chunks from %d indices",
                                len(all_leaf_chunks),
                                sum(len(v) for v in indices_by_job.values()),
                            )
            except Exception:
                logger.warning(
                    "RAPTOR search failed, falling back to leaf search",
                    exc_info=True,
                )

        # Leaf fallback (also used when RAPTOR is disabled)
        if not results:
            if config.enable_raptor:
                logger.info("RAPTOR fallback to leaf search")
            if config.enable_hybrid_search:
                results = await _hybrid_search(query, embedding, db_url, config)
            else:
                results = await search_documents(
                    embedding, db_url, config.rag_max_results, config.rag_max_distance
                )

        if not results:
            logger.info("[tool:retrieve_documents] no results found")
            return "해당 문서에서 관련 내용을 찾지 못했습니다. 웹 검색을 시도해 보세요."

        logger.info(
            "[tool:retrieve_documents] search — strategy=%s, results=%d",
            "raptor"
            if config.enable_raptor
            else ("hybrid" if config.enable_hybrid_search else "dense"),
            len(results),
        )

        # Batch grade documents
        grades = await grade_documents(query, results, config.rag_grading_model)
        relevant_indices = {g.index for g in grades.grades if g.is_relevant}
        relevant_docs = [doc for i, doc in enumerate(results) if i in relevant_indices]
        logger.info(
            "[tool:retrieve_documents] grading — model=%s, total=%d, relevant=%d",
            config.rag_grading_model,
            len(results),
            len(relevant_docs),
        )

        # Rewrite and retry if no relevant docs
        attempts = 0
        while not relevant_docs and attempts < config.rag_max_rewrite_attempts:
            attempts += 1
            rewritten = await rewrite_query(query, config.rag_grading_model)
            logger.info(
                "[tool:retrieve_documents] rewrite attempt %d — original=%r, rewritten=%r",
                attempts,
                query,
                rewritten,
            )
            embedding = await generate_query_embedding(
                rewritten, config.embedding_model
            )
            if config.enable_hybrid_search:
                results = await _hybrid_search(rewritten, embedding, db_url, config)
            else:
                results = await search_documents(
                    embedding, db_url, config.rag_max_results, config.rag_max_distance
                )
            if not results:
                break
            grades = await grade_documents(rewritten, results, config.rag_grading_model)
            relevant_indices = {g.index for g in grades.grades if g.is_relevant}
            relevant_docs = [
                doc for i, doc in enumerate(results) if i in relevant_indices
            ]

        if not relevant_docs:
            logger.info(
                "[tool:retrieve_documents] done — no relevant documents after %d rewrite attempts",
                attempts,
            )
            return "해당 문서에서 관련 내용을 찾지 못했습니다. 웹 검색을 시도해 보세요."

        logger.info(
            "[tool:retrieve_documents] done — returning %d relevant documents (rewrites=%d)",
            len(relevant_docs),
            attempts,
        )
        return await asyncio.to_thread(
            format_results, relevant_docs, config.rag_max_response_tokens
        )

    except Exception:
        logger.exception("Error in retrieve_documents tool")
        return "문서 검색 중 오류가 발생했습니다. 웹 검색을 시도해 보세요."


async def _hybrid_search(
    query: str,
    embedding: list[float],
    db_url: str,
    config: Configuration,
) -> list[dict[str, Any]]:
    """Dense + BM25 하이브리드 검색을 수행한다.

    BM25 실패 시 Dense 결과만 반환한다 (graceful degradation).
    """
    from react_agent.rag import (
        _load_kiwi_config,
        _tokenize_query,
        hybrid_merge,
        search_bm25,
        search_documents,
    )

    # BM25 토큰화 (실패 시 빈 토큰 → Dense only)
    query_tokens: list[str] = []
    try:
        kiwi_config = await _load_kiwi_config(db_url)
        query_tokens = _tokenize_query(
            query,
            kiwi_config["pos_whitelist"],
            kiwi_config["min_keyword_length"],
        )
    except Exception:
        logger.warning("BM25 tokenization failed, using dense only", exc_info=True)

    if not query_tokens:
        return await search_documents(
            embedding, db_url, config.rag_max_results, config.rag_max_distance
        )

    # Dense + BM25 병렬 실행
    dense_results: list[dict[str, Any]]
    sparse_results: list[dict[str, Any]]
    dense_results, sparse_results = await asyncio.gather(
        search_documents(
            embedding, db_url, config.rag_max_results, config.rag_max_distance
        ),
        search_bm25(query_tokens, db_url, config.bm25_top_k),
    )

    if sparse_results:
        logger.info(
            "Hybrid search: %d dense, %d sparse results",
            len(dense_results),
            len(sparse_results),
        )
        return hybrid_merge(dense_results, sparse_results, config.hybrid_alpha)

    logger.info("BM25 returned no results, using dense only")
    return dense_results


TOOLS: List[Callable[..., Any]] = [search, retrieve_documents]
