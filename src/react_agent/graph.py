"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import logging
import os
import re
from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.db import (
    get_database_url,
)
from react_agent.prompts import SUMMARIZATION_PROMPT, TITLE_PROMPT
from react_agent.state import InputState, State
from react_agent.tools import TOOLS

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 16000
"""Max combined character length for summary + retrieved_context in the system prompt."""

_GREETING_PATTERN = re.compile(
    r"^(안녕|hi|hello|hey|thanks|thank you|감사|고마워|bye|ㅎㅇ|ㅂㅂ)[\s!?.]*$",
    re.IGNORECASE,
)

_db_url_initialized = False
_db_url: str | None = None


def _get_db_url() -> str | None:
    """Lazily initialize and cache the database URL."""
    global _db_url_initialized, _db_url  # noqa: PLW0603
    if not _db_url_initialized:
        try:
            _db_url = get_database_url()
        except Exception:
            _db_url = None
        _db_url_initialized = True
    return _db_url


async def summarize_conversation(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Conditionally summarize conversation when message count exceeds threshold.

    If the number of messages exceeds the configured threshold, this node
    summarizes older messages and removes them from state using RemoveMessage.
    The summary is stored in state.summary and prepended to the system prompt.

    Args:
        state: The current conversation state.
        config: Configuration for the model run.

    Returns:
        dict: Empty dict if below threshold, or summary + RemoveMessage list.
    """
    configuration = Configuration.from_runnable_config(config)

    if len(state.messages) <= configuration.summarization_threshold:
        return {}

    # Format messages for summarization (exclude last 2)
    messages_to_summarize = state.messages[:-2]
    conversation_text = "\n".join(
        f"{getattr(m, 'type', 'unknown')}: {getattr(m, 'content', '')}"
        for m in messages_to_summarize
        if getattr(m, "content", "")
    )

    # Call LLM for summarization
    llm = ChatOpenAI(
        temperature=0,
        model=configuration.summarization_model,
    )
    summary_response = await llm.ainvoke(
        SUMMARIZATION_PROMPT.format(conversation=conversation_text)
    )
    summary_content = (
        summary_response.content
        if isinstance(summary_response.content, str)
        else str(summary_response.content)
    )

    # If there's an existing summary, include it for context
    if state.summary:
        summary_content = (
            f"Previous summary: {state.summary}\n\nUpdated summary: {summary_content}"
        )

    # Remove old messages and return new summary
    delete_messages = [
        RemoveMessage(id=m.id) for m in messages_to_summarize if m.id is not None
    ]
    return {"summary": summary_content, "messages": delete_messages}


async def pre_retrieve(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Pre-retrieve documents before calling the model.

    Runs document retrieval on the latest user message so that relevant
    context is available in the system prompt, regardless of LLM tool
    selection. A lightweight gating mechanism skips retrieval for short
    messages, greetings, and when no database is configured.

    Args:
        state: The current conversation state.
        config: Configuration for the model run.

    Returns:
        dict with ``retrieved_context`` key (always present to prevent
        stale context across checkpointer turns).
    """
    # Gate 1: DB availability
    if _get_db_url() is None:
        return {"retrieved_context": ""}

    # Gate 2: Find the last HumanMessage
    human_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not human_msgs:
        return {"retrieved_context": ""}

    query = str(human_msgs[-1].content).strip()

    # Gate 3: Minimum length
    if len(query) < 3:
        return {"retrieved_context": ""}

    # Gate 4: Greeting pattern
    if _GREETING_PATTERN.match(query):
        return {"retrieved_context": ""}

    # Run retrieval
    try:
        from react_agent.tools import retrieve_documents

        result = await retrieve_documents(query)
        if result and "찾지 못했습니다" not in result and "오류" not in result:
            return {"retrieved_context": result}
        return {"retrieved_context": ""}
    except Exception:
        logger.warning("pre_retrieve failed, proceeding without context", exc_info=True)
        return {"retrieved_context": ""}


# Define the function that calls the model


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    # ChatOpenAI 객체 생성
    llm = ChatOpenAI(
        temperature=0.1,
        model=configuration.model,
    )
    model = llm.bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Prepend conversation summary if available
    if state.summary:
        system_message = (
            f"Summary of earlier conversation:\n{state.summary}\n\n{system_message}"
        )

    # Inject pre-retrieved document context
    if state.retrieved_context:
        context = state.retrieved_context
        # Truncate retrieved_context if combined size exceeds budget
        summary_len = len(state.summary) if state.summary else 0
        max_context_len = MAX_CONTEXT_CHARS - summary_len
        if max_context_len > 0 and len(context) > max_context_len:
            context = context[:max_context_len] + "\n\n... (truncated)"
        if max_context_len > 0:
            system_message = (
                f"## Retrieved Document Context\n"
                f"The following documents were retrieved based on the user's query. "
                f"Use this information to answer.\n"
                f"Do not call retrieve_documents again unless the provided context "
                f"is insufficient for answering the question.\n\n"
                f"{context}\n\n{system_message}"
            )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Generate thread title from first user message (once per thread)
    if _get_db_url() and not response.tool_calls:
        try:
            human_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
            if len(human_msgs) == 1:
                configurable = config.get("configurable") or {}
                thread_id = configurable.get("thread_id", "")
                if thread_id:
                    title_llm = ChatOpenAI(
                        temperature=0,
                        model=configuration.summarization_model,
                    )
                    title_response = await title_llm.ainvoke(
                        [{"role": "user", "content": TITLE_PROMPT.format(
                            message=human_msgs[0].content
                        )}]
                    )
                    title = str(title_response.content).strip()[:50]

                    # Write title to LangGraph thread metadata (single source of truth)
                    try:
                        import httpx

                        langgraph_api_url = os.environ.get(
                            "LANGGRAPH_API_URL", "http://localhost:2024"
                        )
                        langgraph_auth_key = os.environ.get("LANGGRAPH_AUTH_KEY", "")
                        headers: dict[str, str] = {}
                        if langgraph_auth_key:
                            headers["Authorization"] = f"Bearer {langgraph_auth_key}"

                        async with httpx.AsyncClient() as http_client:
                            await http_client.patch(
                                f"{langgraph_api_url}/threads/{thread_id}",
                                json={"metadata": {"title": title}},
                                headers=headers,
                                timeout=5.0,
                            )
                    except Exception:
                        logger.exception("Failed to update LangGraph thread metadata")
        except Exception:
            logger.exception("Failed to generate thread title")

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


def create_graph(
    checkpointer: "BaseCheckpointSaver[Any] | None" = None,
) -> Any:
    """Create a compiled ReAct Agent graph.

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
            When None, the graph runs without persistence (suitable for
            langgraph dev which injects its own checkpointer).

    Returns:
        A compiled LangGraph graph.
    """
    builder = StateGraph(State, input=InputState, config_schema=Configuration)

    builder.add_node(summarize_conversation)
    builder.add_node(pre_retrieve)
    builder.add_node(call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    # pre_retrieve must run after summarize_conversation and before call_model
    builder.add_edge("__start__", "summarize_conversation")
    builder.add_edge("summarize_conversation", "pre_retrieve")
    builder.add_edge("pre_retrieve", "call_model")

    builder.add_conditional_edges(
        "call_model",
        route_model_output,
    )

    builder.add_edge("tools", "call_model")

    return builder.compile(checkpointer=checkpointer, name="ReAct Agent")


# Module-level graph for langgraph.json backward compatibility
graph = create_graph()
