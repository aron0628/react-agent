"""Unit tests for the RAG (Retrieval-Augmented Generation) module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from react_agent.rag import (
    DocumentGrade,
    GradeDocuments,
    format_results,
    search_leaf_chunks_by_indices,
    search_raptor_summaries,
)


def test_grade_documents_batch_schema():
    """Verify GradeDocuments / DocumentGrade models accept valid input."""
    grade = DocumentGrade(index=0, is_relevant=True, reasoning="Matches query topic")
    assert grade.index == 0
    assert grade.is_relevant is True

    grades = GradeDocuments(
        grades=[
            DocumentGrade(index=0, is_relevant=True, reasoning="Relevant"),
            DocumentGrade(index=1, is_relevant=False, reasoning="Off topic"),
        ]
    )
    assert len(grades.grades) == 2
    assert grades.grades[0].is_relevant is True
    assert grades.grades[1].is_relevant is False


def test_grade_documents_batch_schema_rejects_invalid():
    """Verify DocumentGrade rejects missing required fields."""
    with pytest.raises(Exception):
        DocumentGrade(index=0, is_relevant=True)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_search_documents_sql_and_joins():
    """Verify search_documents executes correct SQL with JOIN and params."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("psycopg.AsyncConnection.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_conn

        from react_agent.rag import search_documents

        result = await search_documents(
            query_embedding=[0.1, 0.2, 0.3],
            db_url="postgresql://user:pass@localhost/db",
            top_k=5,
            max_distance=0.5,
        )

        assert result == []
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        # Verify SQL structure
        assert "document_embeddings de" in sql
        assert "JOIN parse_jobs pj ON de.job_id = pj.parser_job_id" in sql
        assert "JOIN files f ON pj.file_id = f.id" in sql
        assert "de.embedding <=> %s::vector" in sql
        assert "element_type IN ('text', 'table')" in sql
        assert "ORDER BY distance ASC" in sql

        # Verify params: (embedding_str, embedding_str, max_distance, top_k)
        assert len(params) == 4
        assert params[0] == str([0.1, 0.2, 0.3])
        assert params[1] == str([0.1, 0.2, 0.3])
        assert params[2] == 0.5
        assert params[3] == 5


@pytest.mark.asyncio
async def test_retrieve_no_results_returns_fallback():
    """Verify empty search results return a fallback message."""
    with (
        patch(
            "react_agent.configuration.Configuration.from_context"
        ) as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_config.return_value = _make_config()

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert "찾지 못했습니다" in result


@pytest.mark.asyncio
async def test_retrieve_with_batch_grading_filters():
    """Verify only relevant documents are returned after batch grading."""
    sample_docs = [
        {"content": "Relevant doc", "filename": "a.pdf", "page": 1, "distance": 0.1},
        {"content": "Irrelevant doc", "filename": "b.pdf", "page": 2, "distance": 0.3},
    ]
    grades = GradeDocuments(
        grades=[
            DocumentGrade(index=0, is_relevant=True, reasoning="Matches"),
            DocumentGrade(index=1, is_relevant=False, reasoning="Off topic"),
        ]
    )

    with (
        patch(
            "react_agent.configuration.Configuration.from_context"
        ) as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
            return_value=sample_docs,
        ),
        patch(
            "react_agent.rag.grade_documents",
            new_callable=AsyncMock,
            return_value=grades,
        ),
    ):
        mock_config.return_value = _make_config()

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert "Relevant doc" in result
        assert "Irrelevant doc" not in result


@pytest.mark.asyncio
async def test_rewrite_bounded_by_max_attempts():
    """Verify rewrite loop respects rag_max_rewrite_attempts."""
    empty_grades = GradeDocuments(
        grades=[DocumentGrade(index=0, is_relevant=False, reasoning="No")]
    )
    sample_docs = [
        {"content": "Some doc", "filename": "a.pdf", "page": 1, "distance": 0.2}
    ]

    rewrite_call_count = 0

    async def mock_rewrite(query, model):
        nonlocal rewrite_call_count
        rewrite_call_count += 1
        return f"rewritten: {query}"

    with (
        patch(
            "react_agent.configuration.Configuration.from_context"
        ) as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
            return_value=sample_docs,
        ),
        patch(
            "react_agent.rag.grade_documents",
            new_callable=AsyncMock,
            return_value=empty_grades,
        ),
        patch("react_agent.rag.rewrite_query", side_effect=mock_rewrite),
    ):
        mock_config.return_value = _make_config(rag_max_rewrite_attempts=2)

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert rewrite_call_count == 2
        assert "찾지 못했습니다" in result


def test_format_results_token_budget():
    """Verify format_results truncates at document boundaries."""
    docs = [
        {"content": "Short doc.", "filename": "a.pdf", "page": 1, "distance": 0.1},
        {"content": "X " * 5000, "filename": "b.pdf", "page": 2, "distance": 0.2},
    ]

    result = format_results(docs, max_tokens=50)
    assert "a.pdf" in result
    # Second doc should be excluded (too many tokens)
    assert "b.pdf" not in result


@pytest.mark.asyncio
async def test_retrieve_db_error_returns_message():
    """Verify DB connection failure returns a user-facing error message."""
    with (
        patch(
            "react_agent.configuration.Configuration.from_context"
        ) as mock_config,
        patch(
            "react_agent.db.get_database_url",
            side_effect=RuntimeError("DB_HOST not set"),
        ),
    ):
        mock_config.return_value = _make_config()

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert "오류가 발생했습니다" in result


@pytest.mark.asyncio
async def test_dimension_mismatch_guard():
    """Verify dimension mismatch returns error message instead of crashing."""
    with (
        patch(
            "react_agent.configuration.Configuration.from_context"
        ) as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 1536,  # Wrong dimensions (expected 4096)
        ),
    ):
        mock_config.return_value = _make_config()

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert "차원 불일치" in result


@pytest.mark.asyncio
async def test_search_raptor_summaries_sql():
    """Verify search_raptor_summaries executes correct SQL with params."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("psycopg.AsyncConnection.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_conn

        result = await search_raptor_summaries(
            query_embedding=[0.1, 0.2, 0.3],
            db_url="postgresql://user:pass@localhost/db",
            top_k=3,
            max_distance=0.6,
        )

        assert result == []
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        # Verify SQL structure
        assert "raptor_summaries rs" in sql
        assert "rs.embedding <=> %s::vector" in sql
        assert "rs.job_id" in sql
        assert "rs.raptor_level" in sql
        assert "rs.cluster_id" in sql
        assert "rs.metadata" in sql
        assert "ORDER BY distance ASC" in sql
        assert "LIMIT %s" in sql

        # Verify params
        assert len(params) == 4
        assert params[0] == str([0.1, 0.2, 0.3])
        assert params[1] == str([0.1, 0.2, 0.3])
        assert params[2] == 0.6
        assert params[3] == 3


@pytest.mark.asyncio
async def test_search_leaf_chunks_by_indices_sql():
    """Verify search_leaf_chunks_by_indices uses ANY(%s) and correct JOINs."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("psycopg.AsyncConnection.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_conn

        result = await search_leaf_chunks_by_indices(
            job_id="job-123",
            element_indices=[0, 5, 10],
            db_url="postgresql://user:pass@localhost/db",
        )

        assert result == []
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        # Verify SQL structure
        assert "document_embeddings de" in sql
        assert "JOIN parse_jobs pj ON de.job_id = pj.parser_job_id" in sql
        assert "JOIN files f ON pj.file_id = f.id" in sql
        assert "de.job_id = %s" in sql
        assert "de.element_index = ANY(%s)" in sql
        assert "element_type IN ('text', 'table')" in sql
        assert "ORDER BY de.element_index ASC" in sql

        # Verify params
        assert len(params) == 2
        assert params[0] == "job-123"
        assert params[1] == [0, 5, 10]


@pytest.mark.asyncio
async def test_retrieve_raptor_enabled_clusters_found():
    """When RAPTOR is enabled and clusters found, fetch leaf chunks by indices."""
    raptor_clusters = [
        {
            "job_id": "job-1",
            "raptor_level": 1,
            "cluster_id": 0,
            "content": "Summary of cluster",
            "metadata": {"source_indices": [0, 1, 2], "source_count": 3},
            "distance": 0.3,
        }
    ]
    leaf_chunks = [
        {"content": "Leaf doc A", "filename": "a.pdf", "page": 1, "distance": 0.0},
        {"content": "Leaf doc B", "filename": "a.pdf", "page": 2, "distance": 0.0},
    ]
    grades = GradeDocuments(
        grades=[
            DocumentGrade(index=0, is_relevant=True, reasoning="Matches"),
            DocumentGrade(index=1, is_relevant=True, reasoning="Matches"),
        ]
    )

    with (
        patch("react_agent.configuration.Configuration.from_context") as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_raptor_summaries",
            new_callable=AsyncMock,
            return_value=raptor_clusters,
        ) as mock_raptor,
        patch(
            "react_agent.rag.search_leaf_chunks_by_indices",
            new_callable=AsyncMock,
            return_value=leaf_chunks,
        ) as mock_leaf_by_idx,
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
        ) as mock_search_docs,
        patch(
            "react_agent.rag.grade_documents",
            new_callable=AsyncMock,
            return_value=grades,
        ),
    ):
        mock_config.return_value = _make_config(enable_raptor=True)

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert "Leaf doc A" in result
        mock_raptor.assert_called_once()
        mock_leaf_by_idx.assert_called_once()
        mock_search_docs.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_raptor_enabled_no_clusters_fallback():
    """When RAPTOR finds no clusters, fall back to leaf search."""
    leaf_docs = [
        {"content": "Fallback doc", "filename": "b.pdf", "page": 1, "distance": 0.2},
    ]
    grades = GradeDocuments(
        grades=[DocumentGrade(index=0, is_relevant=True, reasoning="Relevant")]
    )

    with (
        patch("react_agent.configuration.Configuration.from_context") as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_raptor_summaries",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_raptor,
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
            return_value=leaf_docs,
        ) as mock_search_docs,
        patch(
            "react_agent.rag.grade_documents",
            new_callable=AsyncMock,
            return_value=grades,
        ),
    ):
        mock_config.return_value = _make_config(enable_raptor=True)

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert "Fallback doc" in result
        mock_raptor.assert_called_once()
        mock_search_docs.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_raptor_disabled_skips_raptor():
    """When RAPTOR is disabled, search_raptor_summaries is never called."""
    leaf_docs = [
        {"content": "Leaf only", "filename": "c.pdf", "page": 1, "distance": 0.1},
    ]
    grades = GradeDocuments(
        grades=[DocumentGrade(index=0, is_relevant=True, reasoning="Relevant")]
    )

    with (
        patch("react_agent.configuration.Configuration.from_context") as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_raptor_summaries",
            new_callable=AsyncMock,
        ) as mock_raptor,
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
            return_value=leaf_docs,
        ),
        patch(
            "react_agent.rag.grade_documents",
            new_callable=AsyncMock,
            return_value=grades,
        ),
    ):
        mock_config.return_value = _make_config(enable_raptor=False)

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert "Leaf only" in result
        mock_raptor.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_raptor_multiple_clusters_deduplicates():
    """Two clusters with overlapping source_indices should deduplicate."""
    raptor_clusters = [
        {
            "job_id": "job-1",
            "raptor_level": 1,
            "cluster_id": 0,
            "content": "Summary A",
            "metadata": {"source_indices": [0, 1, 2]},
            "distance": 0.3,
        },
        {
            "job_id": "job-1",
            "raptor_level": 1,
            "cluster_id": 1,
            "content": "Summary B",
            "metadata": {"source_indices": [1, 2, 3]},
            "distance": 0.4,
        },
    ]
    leaf_chunks = [
        {"content": "Doc 0", "filename": "a.pdf", "page": 1, "distance": 0.0},
        {"content": "Doc 1", "filename": "a.pdf", "page": 1, "distance": 0.0},
        {"content": "Doc 2", "filename": "a.pdf", "page": 2, "distance": 0.0},
        {"content": "Doc 3", "filename": "a.pdf", "page": 2, "distance": 0.0},
    ]
    grades = GradeDocuments(
        grades=[
            DocumentGrade(index=i, is_relevant=True, reasoning="Match")
            for i in range(4)
        ]
    )

    with (
        patch("react_agent.configuration.Configuration.from_context") as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_raptor_summaries",
            new_callable=AsyncMock,
            return_value=raptor_clusters,
        ),
        patch(
            "react_agent.rag.search_leaf_chunks_by_indices",
            new_callable=AsyncMock,
            return_value=leaf_chunks,
        ) as mock_leaf_by_idx,
        patch(
            "react_agent.rag.grade_documents",
            new_callable=AsyncMock,
            return_value=grades,
        ),
    ):
        mock_config.return_value = _make_config(enable_raptor=True)

        from react_agent.tools import retrieve_documents

        await retrieve_documents("test query")

        # Verify deduplicated indices: {0,1,2} | {1,2,3} = {0,1,2,3}
        call_args = mock_leaf_by_idx.call_args
        passed_indices = call_args[0][1]  # second positional arg
        assert sorted(passed_indices) == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_retrieve_raptor_malformed_metadata_skips():
    """Clusters with missing/malformed metadata.source_indices are skipped."""
    raptor_clusters = [
        {
            "job_id": "job-1",
            "raptor_level": 1,
            "cluster_id": 0,
            "content": "Bad cluster",
            "metadata": None,  # malformed
            "distance": 0.3,
        },
        {
            "job_id": "job-1",
            "raptor_level": 1,
            "cluster_id": 1,
            "content": "Also bad",
            "metadata": {"no_source_indices_key": True},  # missing key
            "distance": 0.4,
        },
    ]
    leaf_docs = [
        {"content": "Fallback leaf", "filename": "d.pdf", "page": 1, "distance": 0.2},
    ]
    grades = GradeDocuments(
        grades=[DocumentGrade(index=0, is_relevant=True, reasoning="Match")]
    )

    with (
        patch("react_agent.configuration.Configuration.from_context") as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_raptor_summaries",
            new_callable=AsyncMock,
            return_value=raptor_clusters,
        ),
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
            return_value=leaf_docs,
        ) as mock_search_docs,
        patch(
            "react_agent.rag.grade_documents",
            new_callable=AsyncMock,
            return_value=grades,
        ),
    ):
        mock_config.return_value = _make_config(enable_raptor=True)

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        # Should fall back to leaf search since all clusters had bad metadata
        assert "Fallback leaf" in result
        mock_search_docs.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_raptor_exception_falls_back():
    """RAPTOR search exception triggers fallback to leaf search."""
    leaf_docs = [
        {"content": "Exception fallback", "filename": "e.pdf", "page": 1, "distance": 0.1},
    ]
    grades = GradeDocuments(
        grades=[DocumentGrade(index=0, is_relevant=True, reasoning="Match")]
    )

    with (
        patch("react_agent.configuration.Configuration.from_context") as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_raptor_summaries",
            new_callable=AsyncMock,
            side_effect=RuntimeError("raptor_summaries table does not exist"),
        ),
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
            return_value=leaf_docs,
        ) as mock_search_docs,
        patch(
            "react_agent.rag.grade_documents",
            new_callable=AsyncMock,
            return_value=grades,
        ),
    ):
        mock_config.return_value = _make_config(enable_raptor=True)

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        assert "Exception fallback" in result
        mock_search_docs.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_raptor_rewrite_skips_raptor():
    """On rewrite retry, RAPTOR should NOT be called again (leaf-only retry)."""
    raptor_clusters = [
        {
            "job_id": "job-1",
            "raptor_level": 1,
            "cluster_id": 0,
            "content": "Summary",
            "metadata": {"source_indices": [0]},
            "distance": 0.3,
        }
    ]
    leaf_from_raptor = [
        {"content": "RAPTOR leaf", "filename": "a.pdf", "page": 1, "distance": 0.0},
    ]
    # First grading: nothing relevant (triggers rewrite)
    empty_grades = GradeDocuments(
        grades=[DocumentGrade(index=0, is_relevant=False, reasoning="No")]
    )
    # After rewrite: relevant
    rewrite_docs = [
        {"content": "Rewrite result", "filename": "b.pdf", "page": 1, "distance": 0.1},
    ]
    good_grades = GradeDocuments(
        grades=[DocumentGrade(index=0, is_relevant=True, reasoning="Yes")]
    )

    grade_call_count = 0

    async def mock_grade(query, docs, model):
        nonlocal grade_call_count
        grade_call_count += 1
        if grade_call_count == 1:
            return empty_grades
        return good_grades

    with (
        patch("react_agent.configuration.Configuration.from_context") as mock_config,
        patch("react_agent.db.get_database_url", return_value="postgresql://x"),
        patch(
            "react_agent.rag.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 4096,
        ),
        patch(
            "react_agent.rag.search_raptor_summaries",
            new_callable=AsyncMock,
            return_value=raptor_clusters,
        ) as mock_raptor,
        patch(
            "react_agent.rag.search_leaf_chunks_by_indices",
            new_callable=AsyncMock,
            return_value=leaf_from_raptor,
        ),
        patch(
            "react_agent.rag.search_documents",
            new_callable=AsyncMock,
            return_value=rewrite_docs,
        ),
        patch("react_agent.rag.grade_documents", side_effect=mock_grade),
        patch(
            "react_agent.rag.rewrite_query",
            new_callable=AsyncMock,
            return_value="rewritten query",
        ),
    ):
        mock_config.return_value = _make_config(
            enable_raptor=True, rag_max_rewrite_attempts=1
        )

        from react_agent.tools import retrieve_documents

        result = await retrieve_documents("test query")
        # RAPTOR should be called exactly once (initial), not on rewrite
        assert mock_raptor.call_count == 1
        assert "Rewrite result" in result


def _make_config(**overrides):
    """Create a Configuration with default RAG settings."""
    defaults = {
        "system_prompt": "test",
        "model": "gpt-4.1",
        "max_search_results": 10,
        "summarization_threshold": 20,
        "summarization_model": "gpt-4.1-mini",
        "embedding_model": "solar-embedding-1-large",
        "embedding_dimensions": 4096,
        "rag_max_distance": 0.5,
        "rag_max_results": 5,
        "rag_max_rewrite_attempts": 1,
        "rag_grading_model": "gpt-4.1-mini",
        "rag_max_response_tokens": 4000,
        "enable_raptor": False,
        "raptor_max_distance": 0.6,
        "raptor_top_k": 3,
    }
    defaults.update(overrides)

    from react_agent.configuration import Configuration

    return Configuration(**defaults)
