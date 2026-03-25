"""Tests for pre_retrieve node and retrieved_context injection."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from react_agent.graph import MAX_CONTEXT_CHARS, pre_retrieve
from react_agent.state import State

# ---------------------------------------------------------------------------
# pre_retrieve node tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_retrieve_with_human_message_and_db():
    """Should call retrieve_documents and return result when DB is available."""
    state = State(messages=[HumanMessage(content="삼성전자 생성형AI 설명해줘", id="1")])
    config = {"configurable": {}}

    with (
        patch(
            "react_agent.graph._get_db_url", return_value="postgresql://localhost/test"
        ),
        patch(
            "react_agent.tools.retrieve_documents",
            new_callable=AsyncMock,
            return_value="문서 결과: 삼성전자는 ...",
        ) as mock_retrieve,
    ):
        result = await pre_retrieve(state, config)

    mock_retrieve.assert_awaited_once_with("삼성전자 생성형AI 설명해줘")
    assert result["retrieved_context"] == "문서 결과: 삼성전자는 ..."


@pytest.mark.asyncio
async def test_pre_retrieve_no_human_message():
    """Should return empty context when no HumanMessage exists."""
    state = State(messages=[AIMessage(content="Hello", id="1")])
    config = {"configurable": {}}

    with patch(
        "react_agent.graph._get_db_url", return_value="postgresql://localhost/test"
    ):
        result = await pre_retrieve(state, config)

    assert result == {"retrieved_context": ""}


@pytest.mark.asyncio
async def test_pre_retrieve_no_db():
    """Should return empty context when DB URL is None."""
    state = State(messages=[HumanMessage(content="테스트 질문", id="1")])
    config = {"configurable": {}}

    with patch("react_agent.graph._get_db_url", return_value=None):
        result = await pre_retrieve(state, config)

    assert result == {"retrieved_context": ""}


@pytest.mark.asyncio
async def test_pre_retrieve_exception_graceful():
    """Should return empty context and log when retrieve_documents raises."""
    state = State(messages=[HumanMessage(content="테스트 질문입니다", id="1")])
    config = {"configurable": {}}

    with (
        patch(
            "react_agent.graph._get_db_url", return_value="postgresql://localhost/test"
        ),
        patch(
            "react_agent.tools.retrieve_documents",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB connection failed"),
        ),
    ):
        result = await pre_retrieve(state, config)

    assert result == {"retrieved_context": ""}


@pytest.mark.asyncio
async def test_pre_retrieve_short_query_skip():
    """Should skip retrieval for very short messages (< 3 chars)."""
    state = State(messages=[HumanMessage(content="ㅇㅋ", id="1")])
    config = {"configurable": {}}

    with patch(
        "react_agent.graph._get_db_url", return_value="postgresql://localhost/test"
    ):
        result = await pre_retrieve(state, config)

    assert result == {"retrieved_context": ""}


@pytest.mark.asyncio
async def test_pre_retrieve_greeting_skip():
    """Should skip retrieval for greeting messages."""
    config = {"configurable": {}}

    greetings = ["hello", "안녕", "Hi!", "thanks!", "감사", "hey", "bye"]
    for greeting in greetings:
        state = State(messages=[HumanMessage(content=greeting, id="1")])

        with patch(
            "react_agent.graph._get_db_url", return_value="postgresql://localhost/test"
        ):
            result = await pre_retrieve(state, config)

        assert result == {"retrieved_context": ""}, f"Failed for greeting: {greeting}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "no_db",
        "no_human_message",
        "short_query",
        "greeting",
        "exception",
        "no_results",
        "valid_results",
    ],
)
async def test_pre_retrieve_always_returns_retrieved_context_key(scenario):
    """Every code path must return a dict with 'retrieved_context' key."""
    config = {"configurable": {}}

    if scenario == "no_db":
        state = State(messages=[HumanMessage(content="질문", id="1")])
        with patch("react_agent.graph._get_db_url", return_value=None):
            result = await pre_retrieve(state, config)
    elif scenario == "no_human_message":
        state = State(messages=[AIMessage(content="Hi", id="1")])
        with patch("react_agent.graph._get_db_url", return_value="pg://test"):
            result = await pre_retrieve(state, config)
    elif scenario == "short_query":
        state = State(messages=[HumanMessage(content="ab", id="1")])
        with patch("react_agent.graph._get_db_url", return_value="pg://test"):
            result = await pre_retrieve(state, config)
    elif scenario == "greeting":
        state = State(messages=[HumanMessage(content="hello", id="1")])
        with patch("react_agent.graph._get_db_url", return_value="pg://test"):
            result = await pre_retrieve(state, config)
    elif scenario == "exception":
        state = State(messages=[HumanMessage(content="긴 질문입니다", id="1")])
        with (
            patch("react_agent.graph._get_db_url", return_value="pg://test"),
            patch(
                "react_agent.tools.retrieve_documents",
                new_callable=AsyncMock,
                side_effect=Exception("fail"),
            ),
        ):
            result = await pre_retrieve(state, config)
    elif scenario == "no_results":
        state = State(messages=[HumanMessage(content="긴 질문입니다", id="1")])
        with (
            patch("react_agent.graph._get_db_url", return_value="pg://test"),
            patch(
                "react_agent.tools.retrieve_documents",
                new_callable=AsyncMock,
                return_value="해당 문서에서 관련 내용을 찾지 못했습니다.",
            ),
        ):
            result = await pre_retrieve(state, config)
    else:  # valid_results
        state = State(messages=[HumanMessage(content="긴 질문입니다", id="1")])
        with (
            patch("react_agent.graph._get_db_url", return_value="pg://test"),
            patch(
                "react_agent.tools.retrieve_documents",
                new_callable=AsyncMock,
                return_value="문서 결과입니다",
            ),
        ):
            result = await pre_retrieve(state, config)

    assert "retrieved_context" in result, f"Missing key for scenario: {scenario}"
    assert isinstance(result["retrieved_context"], str)


@pytest.mark.asyncio
async def test_pre_retrieve_config_from_context_works_in_node():
    """Verify Configuration.from_context() resolves in node execution context."""
    state = State(messages=[HumanMessage(content="테스트 질문입니다", id="1")])
    config = {"configurable": {"embedding_model": "test-model"}}

    with (
        patch("react_agent.graph._get_db_url", return_value="pg://test"),
        patch(
            "react_agent.tools.retrieve_documents",
            new_callable=AsyncMock,
            return_value="결과",
        ) as mock_retrieve,
    ):
        result = await pre_retrieve(state, config)

    mock_retrieve.assert_awaited_once()
    assert result["retrieved_context"] == "결과"


# ---------------------------------------------------------------------------
# call_model retrieved_context injection tests
# ---------------------------------------------------------------------------


def _make_call_model_mocks():
    """Create standard mocks for call_model tests."""
    mock_model = AsyncMock()
    mock_response = AIMessage(content="Response", id="resp1")
    mock_model.ainvoke.return_value = mock_response
    mock_llm_instance = MagicMock()
    mock_llm_instance.bind_tools.return_value = mock_model
    return mock_llm_instance, mock_model


@pytest.mark.asyncio
async def test_call_model_includes_retrieved_context():
    """System prompt should include retrieved context when available."""
    from react_agent.graph import call_model

    state = State(
        messages=[HumanMessage(content="Hello", id="1")],
        retrieved_context="문서 내용: 삼성전자는 생성형AI를 개발했습니다.",
    )
    config = {"configurable": {}}

    with patch("react_agent.graph.load_chat_model") as mock_llm_cls:
        mock_llm_instance, mock_model = _make_call_model_mocks()
        mock_llm_cls.return_value = mock_llm_instance

        await call_model(state, config)

    call_args = mock_model.ainvoke.call_args[0][0]
    system_msg = call_args[0]["content"]
    assert "Retrieved Document Context" in system_msg
    assert "삼성전자는 생성형AI를 개발했습니다" in system_msg


@pytest.mark.asyncio
async def test_call_model_retrieved_context_includes_dedup_instruction():
    """System prompt should include dedup instruction when context is present."""
    from react_agent.graph import call_model

    state = State(
        messages=[HumanMessage(content="Hello", id="1")],
        retrieved_context="Some document content",
    )
    config = {"configurable": {}}

    with patch("react_agent.graph.load_chat_model") as mock_llm_cls:
        mock_llm_instance, mock_model = _make_call_model_mocks()
        mock_llm_cls.return_value = mock_llm_instance

        await call_model(state, config)

    call_args = mock_model.ainvoke.call_args[0][0]
    system_msg = call_args[0]["content"]
    assert "Do not call retrieve_documents again" in system_msg


@pytest.mark.asyncio
async def test_call_model_without_retrieved_context():
    """System prompt should not change when retrieved_context is empty."""
    from react_agent.graph import call_model

    state = State(
        messages=[HumanMessage(content="Hello", id="1")],
        retrieved_context="",
    )
    config = {"configurable": {}}

    with patch("react_agent.graph.load_chat_model") as mock_llm_cls:
        mock_llm_instance, mock_model = _make_call_model_mocks()
        mock_llm_cls.return_value = mock_llm_instance

        await call_model(state, config)

    call_args = mock_model.ainvoke.call_args[0][0]
    system_msg = call_args[0]["content"]
    assert "Retrieved Document Context" not in system_msg


@pytest.mark.asyncio
async def test_call_model_truncates_long_context():
    """Should truncate retrieved_context when combined size exceeds MAX_CONTEXT_CHARS."""
    from react_agent.graph import call_model

    long_context = "A" * (MAX_CONTEXT_CHARS + 1000)
    state = State(
        messages=[HumanMessage(content="Hello", id="1")],
        retrieved_context=long_context,
    )
    config = {"configurable": {}}

    with patch("react_agent.graph.load_chat_model") as mock_llm_cls:
        mock_llm_instance, mock_model = _make_call_model_mocks()
        mock_llm_cls.return_value = mock_llm_instance

        await call_model(state, config)

    call_args = mock_model.ainvoke.call_args[0][0]
    system_msg = call_args[0]["content"]
    assert "... (truncated)" in system_msg
    # The injected context should be shorter than the original
    assert len(long_context) not in [len(system_msg)]


@pytest.mark.asyncio
async def test_call_model_truncate_preserves_summary():
    """Summary should be preserved when truncation is needed."""
    from react_agent.graph import call_model

    summary = "This is an important conversation summary."
    long_context = "B" * (MAX_CONTEXT_CHARS + 1000)
    state = State(
        messages=[HumanMessage(content="Hello", id="1")],
        summary=summary,
        retrieved_context=long_context,
    )
    config = {"configurable": {}}

    with patch("react_agent.graph.load_chat_model") as mock_llm_cls:
        mock_llm_instance, mock_model = _make_call_model_mocks()
        mock_llm_cls.return_value = mock_llm_instance

        await call_model(state, config)

    call_args = mock_model.ainvoke.call_args[0][0]
    system_msg = call_args[0]["content"]
    # Summary must be fully preserved
    assert summary in system_msg
    # Context should be truncated
    assert "... (truncated)" in system_msg
