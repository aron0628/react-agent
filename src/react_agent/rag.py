"""RAG (Retrieval-Augmented Generation) module for document search and grading.

Provides vector similarity search against the document_embeddings table,
batch document grading via structured LLM output, query rewriting,
and token-budgeted result formatting.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
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


# ---------------------------------------------------------------------------
# BM25 하이브리드 검색 모듈-레벨 싱글턴
# ---------------------------------------------------------------------------

_kiwi: Any = None
_stopwords: set[str] = set()
_kiwi_config_cache: dict = {}
_kiwi_config_loaded_at: float = 0


def _init_kiwi() -> Any:
    """Kiwi 형태소 분석기 싱글턴을 초기화하고 반환한다."""
    global _kiwi, _stopwords  # noqa: PLW0603

    if _kiwi is not None:
        return _kiwi

    from kiwipiepy import Kiwi

    _kiwi = Kiwi()

    # 불용어 로드
    stopwords_path = Path(__file__).parent / "resources" / "korean_stopwords.txt"
    try:
        with open(stopwords_path, encoding="utf-8") as f:
            _stopwords = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logger.warning(f"불용어 파일을 찾을 수 없음: {stopwords_path}")
        _stopwords = set()

    return _kiwi


async def _load_kiwi_config(db_url: str) -> dict:
    """keyword_config 테이블에서 POS 화이트리스트와 최소 키워드 길이를 로드한다.

    5분간 캐시하여 반복 DB 조회를 방지한다.
    실패 시 기본값을 반환한다.
    """
    global _kiwi_config_cache, _kiwi_config_loaded_at  # noqa: PLW0603

    now = time.time()
    if _kiwi_config_cache and (now - _kiwi_config_loaded_at) < 300:
        return _kiwi_config_cache

    defaults = {
        "pos_whitelist": ["NNG", "NNP", "SL", "SH"],
        "min_keyword_length": 2,
    }

    try:
        import psycopg
        from psycopg.rows import dict_row  # type: ignore[import-not-found]

        async with await psycopg.AsyncConnection.connect(
            db_url, row_factory=dict_row
        ) as conn:
            cursor = await conn.execute(
                "SELECT pos_whitelist, min_keyword_length FROM keyword_config LIMIT 1"
            )
            row = await cursor.fetchone()
            if row:
                _kiwi_config_cache = {
                    "pos_whitelist": row["pos_whitelist"],
                    "min_keyword_length": row["min_keyword_length"],
                }
            else:
                _kiwi_config_cache = defaults
    except Exception as e:
        logger.warning(f"keyword_config 로드 실패, 기본값 사용: {e}")
        _kiwi_config_cache = defaults

    _kiwi_config_loaded_at = now
    return _kiwi_config_cache


def _tokenize_query(
    query: str, pos_whitelist: list[str], min_length: int
) -> list[str]:
    """쿼리 텍스트를 형태소 분석하여 검색용 키워드 리스트를 반환한다.

    2-stage 필터링:
      Stage 1: Kiwi 형태소 분석 → POS 화이트리스트 필터
      Stage 2: 불용어 제거 + 최소 길이 필터
    """
    kiwi = _init_kiwi()

    # Stage 1: 형태소 분석 → POS 화이트리스트 필터
    tokens = kiwi.tokenize(query)
    candidates = [token.form for token in tokens if token.tag in pos_whitelist]

    # Stage 2: 불용어 제거 + 최소 길이 필터
    keywords = [
        w.lower()
        for w in candidates
        if len(w) >= min_length and w.lower() not in _stopwords
    ]
    return keywords


async def search_bm25(
    query_tokens: list[str],
    db_url: str,
    top_k: int = 20,
) -> list[dict]:
    """BM25 키워드 검색을 수행한다.

    document_keywords 테이블에서 쿼리 토큰과 매칭되는 문서를 검색하고,
    간소화된 BM25 점수(IDF * TF)를 계산하여 상위 결과를 반환한다.
    실패 시 빈 리스트를 반환한다 (graceful degradation).

    Args:
        query_tokens: 형태소 분석된 쿼리 키워드 리스트.
        db_url: PostgreSQL 연결 URL.
        top_k: 반환할 최대 결과 수.

    Returns:
        BM25 점수 기준 내림차순 정렬된 문서 딕셔너리 리스트.
    """
    if not query_tokens:
        return []

    try:
        import json

        import psycopg
        from psycopg.rows import dict_row  # type: ignore[import-not-found]

        async with await psycopg.AsyncConnection.connect(
            db_url, row_factory=dict_row
        ) as conn:
            # 전체 문서 수 조회 (IDF 계산용)
            cursor = await conn.execute(
                "SELECT COUNT(*) AS cnt FROM document_keywords"
            )
            row = await cursor.fetchone()
            total_docs = row["cnt"] if row else 0

            if total_docs == 0:
                return []

            # 쿼리 토큰별 DF (Document Frequency) 조회
            df_map: dict[str, int] = {}
            for token in query_tokens:
                cursor = await conn.execute(
                    "SELECT COUNT(*) AS cnt FROM document_keywords "
                    "WHERE %s = ANY(keywords)",
                    (token,),
                )
                df_row = await cursor.fetchone()
                df_map[token] = df_row["cnt"] if df_row else 0

            # 매칭 문서 조회
            cursor = await conn.execute(
                "SELECT dk.job_id, dk.element_index, dk.page, "
                "dk.keywords, dk.tf_scores, "
                "de.content, de.metadata "
                "FROM document_keywords dk "
                "JOIN document_embeddings de ON dk.job_id = de.job_id "
                "AND dk.element_index = de.parent_element_index "
                "WHERE dk.keywords && %s::text[]",
                (query_tokens,),
            )
            rows = await cursor.fetchall()

            # BM25 점수 계산
            results: list[dict] = []
            for row in rows:
                tf_scores = row["tf_scores"]
                if isinstance(tf_scores, str):
                    tf_scores = json.loads(tf_scores)

                score = 0.0
                for token in query_tokens:
                    df = df_map.get(token, 0)
                    if df == 0:
                        continue
                    idf = math.log(total_docs / df)
                    tf = tf_scores.get(token, 0.0)
                    score += idf * tf

                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                results.append({
                    "content": row["content"],
                    "metadata": metadata,
                    "page": row["page"],
                    "job_id": row["job_id"],
                    "element_index": row["element_index"],
                    "bm25_score": score,
                })

            # 점수 내림차순 정렬 후 top_k 제한
            results.sort(key=lambda x: x["bm25_score"], reverse=True)
            return results[:top_k]

    except Exception as e:
        logger.error(f"BM25 검색 실패: {e}")
        return []


def hybrid_merge(
    dense_results: list[dict],
    sparse_results: list[dict],
    alpha: float = 0.7,
) -> list[dict]:
    """Dense(벡터) 검색과 Sparse(BM25) 검색 결과를 하이브리드 병합한다.

    Min-max 정규화 후 가중합으로 최종 점수를 산출한다.

    Args:
        dense_results: 벡터 검색 결과 (distance 필드 포함).
        sparse_results: BM25 검색 결과 (bm25_score 필드 포함).
        alpha: Dense 가중치 (0.0=순수 Sparse, 1.0=순수 Dense).

    Returns:
        hybrid_score 기준 내림차순 정렬된 병합 결과 리스트.
    """
    # Dense 점수 정규화: distance → similarity (1 - distance) → min-max
    dense_scores: dict[tuple[str, int], float] = {}
    if dense_results:
        similarities = [1.0 - d.get("distance", 0.0) for d in dense_results]
        sim_min = min(similarities)
        sim_max = max(similarities)
        sim_range = sim_max - sim_min if sim_max != sim_min else 1.0

        for doc, sim in zip(dense_results, similarities):
            key = (doc.get("job_id", ""), doc.get("element_index", 0))
            dense_scores[key] = (sim - sim_min) / sim_range

    # Sparse 점수 정규화: bm25_score → min-max
    sparse_scores: dict[tuple[str, int], float] = {}
    if sparse_results:
        bm25_vals = [d.get("bm25_score", 0.0) for d in sparse_results]
        bm25_min = min(bm25_vals)
        bm25_max = max(bm25_vals)
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

        for doc in sparse_results:
            key = (doc.get("job_id", ""), doc.get("element_index", 0))
            sparse_scores[key] = (doc.get("bm25_score", 0.0) - bm25_min) / bm25_range

    # 모든 고유 키 수집
    all_keys = set(dense_scores.keys()) | set(sparse_scores.keys())

    # 문서 데이터 인덱스 구축
    doc_map: dict[tuple[str, int], dict] = {}
    for doc in dense_results:
        key = (doc.get("job_id", ""), doc.get("element_index", 0))
        doc_map[key] = doc
    for doc in sparse_results:
        key = (doc.get("job_id", ""), doc.get("element_index", 0))
        if key not in doc_map:
            doc_map[key] = doc

    # 하이브리드 점수 계산 및 병합
    merged: list[dict] = []
    for key in all_keys:
        dense_norm = dense_scores.get(key, 0.0)
        sparse_norm = sparse_scores.get(key, 0.0)
        hybrid_score = alpha * dense_norm + (1.0 - alpha) * sparse_norm

        doc = {**doc_map[key], "hybrid_score": hybrid_score}
        merged.append(doc)

    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return merged
