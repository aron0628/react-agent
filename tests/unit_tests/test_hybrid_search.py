"""하이브리드 검색 (Dense + BM25) 테스트"""

import pytest

from react_agent.rag import _tokenize_query, hybrid_merge

# ---------------------------------------------------------------------------
# hybrid_merge 테스트
# ---------------------------------------------------------------------------


def _dense(
    content: str, distance: float, job_id: str, element_index: int, page: int = 1
) -> dict:
    return {
        "content": content,
        "distance": distance,
        "job_id": job_id,
        "element_index": element_index,
        "page": page,
        "metadata": {},
    }


def _sparse(
    content: str, bm25_score: float, job_id: str, element_index: int, page: int = 1
) -> dict:
    return {
        "content": content,
        "bm25_score": bm25_score,
        "job_id": job_id,
        "element_index": element_index,
        "page": page,
        "metadata": {},
    }


def test_hybrid_merge_alpha_1():
    """alpha=1.0 → Dense only: sparse-only result gets score 0, dense result ranks higher."""
    # 두 개의 dense 문서가 있어야 min-max 정규화에 범위가 생김
    dense_results = [
        _dense("a_close", 0.1, "j1", 0),  # similarity=0.9 → normalized=1.0
        _dense("a_far", 0.5, "j1", 2),  # similarity=0.5 → normalized=0.0
    ]
    sparse_results = [_sparse("b", 5.0, "j2", 1)]

    merged = hybrid_merge(dense_results, sparse_results, alpha=1.0)

    # 세 문서 모두 반환
    assert len(merged) == 3

    # 최상위는 dense에서 가장 유사도 높은 "a_close"
    assert merged[0]["content"] == "a_close"

    # sparse-only 문서는 dense 컴포넌트 0.0 → hybrid_score = 0.0
    b_doc = next(d for d in merged if d["content"] == "b")
    assert b_doc["hybrid_score"] == pytest.approx(0.0)


def test_hybrid_merge_alpha_0():
    """alpha=0.0 → Sparse only: dense-only result gets score 0, sparse result ranks higher."""
    dense_results = [_dense("a", 0.2, "j1", 0)]
    # 두 개의 sparse 문서가 있어야 min-max 정규화에 범위가 생김
    sparse_results = [
        _sparse("b_high", 5.0, "j2", 1),  # bm25 높음 → normalized=1.0
        _sparse("b_low", 1.0, "j2", 2),  # bm25 낮음 → normalized=0.0
    ]

    merged = hybrid_merge(dense_results, sparse_results, alpha=0.0)

    assert len(merged) == 3

    # 최상위는 sparse에서 점수 가장 높은 "b_high"
    assert merged[0]["content"] == "b_high"

    # dense-only 문서는 sparse 컴포넌트 0.0 → hybrid_score = 0.0
    a_doc = next(d for d in merged if d["content"] == "a")
    assert a_doc["hybrid_score"] == pytest.approx(0.0)


def test_hybrid_merge_balanced():
    """alpha=0.5: 두 결과 모두에 등장하는 문서가 가장 높은 점수를 받는다."""
    # j1/0은 dense + sparse 모두 존재
    dense_results = [
        _dense("c", 0.3, "j1", 0),
        _dense("d", 0.8, "j2", 1),
    ]
    sparse_results = [
        _sparse("c", 3.0, "j1", 0),
        _sparse("e", 1.0, "j3", 2),
    ]

    merged = hybrid_merge(dense_results, sparse_results, alpha=0.5)

    assert len(merged) == 3  # j1/0, j2/1, j3/2

    # j1/0 (content="c")이 두 검색 모두에서 상위권이므로 최고 점수여야 함
    top = merged[0]
    assert top["content"] == "c"
    assert top["hybrid_score"] > 0.0


def test_hybrid_merge_dedup():
    """동일 문서(job_id + element_index)가 양쪽에 있으면 결과에 1번만 나온다."""
    dense_results = [_dense("dup", 0.1, "j1", 0)]
    sparse_results = [_sparse("dup", 4.0, "j1", 0)]

    merged = hybrid_merge(dense_results, sparse_results, alpha=0.5)

    assert len(merged) == 1
    assert merged[0]["content"] == "dup"
    assert "hybrid_score" in merged[0]


def test_hybrid_merge_empty_sparse():
    """sparse_results=[] → dense 결과만 반환, hybrid_score 포함."""
    dense_results = [
        _dense("x", 0.1, "j1", 0),
        _dense("y", 0.4, "j1", 1),
    ]

    merged = hybrid_merge(dense_results, [], alpha=0.7)

    assert len(merged) == 2
    for doc in merged:
        assert "hybrid_score" in doc

    # distance가 낮을수록(similarity 높을수록) 점수가 높아야 함
    scores = [d["hybrid_score"] for d in merged]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_merge_empty_dense():
    """dense_results=[] → sparse 결과만 반환, hybrid_score 포함."""
    sparse_results = [
        _sparse("p", 6.0, "j1", 0),
        _sparse("q", 2.0, "j1", 1),
    ]

    merged = hybrid_merge([], sparse_results, alpha=0.7)

    assert len(merged) == 2
    for doc in merged:
        assert "hybrid_score" in doc

    # bm25_score가 높은 문서가 상위여야 함
    assert merged[0]["content"] == "p"
    assert merged[1]["content"] == "q"


# ---------------------------------------------------------------------------
# _tokenize_query 테스트 (실제 Kiwi 사용)
# ---------------------------------------------------------------------------

_DEFAULT_POS = ["NNG", "NNP", "SL", "SH"]
_DEFAULT_MIN_LEN = 2


def test_tokenize_query_korean():
    """한국어 쿼리 → 명사 키워드 추출, 조사/불용어 제거."""
    tokens = _tokenize_query(
        "삼성전자 반도체 투자 현황", _DEFAULT_POS, _DEFAULT_MIN_LEN
    )

    assert isinstance(tokens, list)
    assert len(tokens) > 0

    # 핵심 명사가 포함돼야 함
    assert "삼성전자" in tokens
    assert "반도체" in tokens

    # 조사류(JX, JC 등)는 whitelist에 없으므로 없어야 함
    for token in tokens:
        assert len(token) >= _DEFAULT_MIN_LEN


def test_tokenize_query_mixed():
    """영어+한국어 혼합 쿼리: SL 태그 영어 토큰은 소문자로 반환된다."""
    tokens = _tokenize_query("FastAPI 서버 성능", _DEFAULT_POS, _DEFAULT_MIN_LEN)

    assert isinstance(tokens, list)
    # SL(외국어) 태그로 분석된 FastAPI → 소문자 "fastapi"
    assert "fastapi" in tokens


def test_tokenize_query_empty():
    """빈 문자열 입력 → 빈 리스트 반환."""
    tokens = _tokenize_query("", _DEFAULT_POS, _DEFAULT_MIN_LEN)

    assert tokens == []
