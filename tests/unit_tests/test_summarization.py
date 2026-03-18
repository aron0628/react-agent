"""Tests for conversation summarization logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from react_agent.state import State


@pytest.mark.asyncio
async def test_summarize_below_threshold_is_noop():
    """Summarization should return empty dict when below threshold."""
    from react_agent.graph import summarize_conversation

    state = State(
        messages=[
            HumanMessage(content="Hello", id="1"),
            AIMessage(content="Hi there!", id="2"),
        ],
    )
    config = {"configurable": {"summarization_threshold": 20}}
    result = await summarize_conversation(state, config)
    assert result == {}


@pytest.mark.asyncio
async def test_summarize_above_threshold_returns_remove_messages():
    """Summarization should return RemoveMessage for old messages when above threshold."""
    from react_agent.graph import summarize_conversation

    messages = [HumanMessage(content=f"Message {i}", id=str(i)) for i in range(25)]
    state = State(messages=messages)
    config = {"configurable": {"summarization_threshold": 5}}

    with patch("react_agent.graph.ChatOpenAI") as mock_llm_cls:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Summary of conversation")
        mock_llm_cls.return_value = mock_llm

        result = await summarize_conversation(state, config)

    assert "messages" in result
    assert "summary" in result
    # Should remove all but last 2 messages
    remove_msgs = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
    assert len(remove_msgs) == 23


@pytest.mark.asyncio
async def test_summarize_above_threshold_returns_summary():
    """Summarization should return summary content from LLM."""
    from react_agent.graph import summarize_conversation

    messages = [HumanMessage(content=f"Message {i}", id=str(i)) for i in range(10)]
    state = State(messages=messages)
    config = {"configurable": {"summarization_threshold": 3}}

    with patch("react_agent.graph.ChatOpenAI") as mock_llm_cls:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Test summary content")
        mock_llm_cls.return_value = mock_llm

        result = await summarize_conversation(state, config)

    assert "Test summary content" in result["summary"]


@pytest.mark.asyncio
async def test_call_model_includes_summary_in_prompt():
    """call_model should prepend summary to system message when available."""
    from react_agent.graph import call_model

    state = State(
        messages=[HumanMessage(content="Hello", id="1")],
        summary="Previous conversation was about weather.",
    )
    config = {"configurable": {}}

    with patch("react_agent.graph.ChatOpenAI") as mock_llm_cls:
        mock_model = AsyncMock()
        mock_response = AIMessage(content="Response", id="resp1")
        mock_model.ainvoke.return_value = mock_response
        mock_llm_instance = MagicMock()
        mock_llm_instance.bind_tools.return_value = mock_model
        mock_llm_cls.return_value = mock_llm_instance

        await call_model(state, config)

    # Verify the system message contains the summary
    call_args = mock_model.ainvoke.call_args[0][0]
    system_msg = call_args[0]
    assert "Summary of earlier conversation" in system_msg["content"]
    assert "Previous conversation was about weather" in system_msg["content"]


@pytest.mark.asyncio
async def test_call_model_without_summary_has_no_prefix():
    """call_model should not add summary prefix when summary is empty."""
    from react_agent.graph import call_model

    state = State(
        messages=[HumanMessage(content="Hello", id="1")],
    )
    config = {"configurable": {}}

    with patch("react_agent.graph.ChatOpenAI") as mock_llm_cls:
        mock_model = AsyncMock()
        mock_response = AIMessage(content="Response", id="resp1")
        mock_model.ainvoke.return_value = mock_response
        mock_llm_instance = MagicMock()
        mock_llm_instance.bind_tools.return_value = mock_model
        mock_llm_cls.return_value = mock_llm_instance

        await call_model(state, config)

    call_args = mock_model.ainvoke.call_args[0][0]
    system_msg = call_args[0]
    assert "Summary of earlier conversation" not in system_msg["content"]
