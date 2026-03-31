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
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import var_child_runnable_config  # type: ignore[attr-defined]
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.db import (
    ensure_settings_loaded,
    get_database_url,
)
from react_agent.prompts import SUMMARIZATION_PROMPT, TITLE_PROMPT
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_CONTEXT_CHARS = 16000
"""Max combined character length for summary + retrieved_context in the system prompt."""

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

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
    db_url = _get_db_url()
    if db_url:
        await ensure_settings_loaded(db_url)

    configuration = Configuration.from_runnable_config(config)
    msg_count = len(state.messages)

    if msg_count <= configuration.summarization_threshold:
        logger.info(
            "[summarize_conversation] skip — messages=%d, threshold=%d",
            msg_count,
            configuration.summarization_threshold,
        )
        return {}

    logger.info(
        "[summarize_conversation] summarizing — messages=%d, threshold=%d",
        msg_count,
        configuration.summarization_threshold,
    )

    # Format messages for summarization (exclude last 2)
    messages_to_summarize = state.messages[:-2]
    conversation_text = "\n".join(
        f"{getattr(m, 'type', 'unknown')}: {getattr(m, 'content', '')}"
        for m in messages_to_summarize
        if getattr(m, "content", "")
    )

    # Call LLM for summarization
    logger.info("[summarize_conversation] model=%s", configuration.summarization_model)
    llm = load_chat_model(configuration.summarization_model, temperature=0)
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
    logger.info(
        "[summarize_conversation] done — removed %d messages, summary_len=%d",
        len(delete_messages),
        len(summary_content),
    )
    return {"summary": summary_content, "messages": delete_messages}


async def pre_retrieve(state: State, config: RunnableConfig) -> dict[str, Any]:
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
        logger.info("[pre_retrieve] skip — no database configured")
        return {"retrieved_context": ""}

    # Gate 2: Find the last HumanMessage
    human_msgs = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not human_msgs:
        logger.info("[pre_retrieve] skip — no user message found")
        return {"retrieved_context": ""}

    raw_content = human_msgs[-1].content
    if isinstance(raw_content, list):
        query = " ".join(
            block["text"]
            for block in raw_content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    else:
        query = str(raw_content).strip()

    # Gate 3: Minimum length
    if len(query) < 3:
        logger.info("[pre_retrieve] skip — query too short: %r", query)
        return {"retrieved_context": ""}

    # Gate 4: Greeting pattern
    if _GREETING_PATTERN.match(query):
        logger.info("[pre_retrieve] skip — greeting detected: %r", query)
        return {"retrieved_context": ""}

    logger.info("[pre_retrieve] retrieving — query=%r", query)

    # Run retrieval with callbacks disabled to prevent internal LLM
    # calls (grading, rewrite) from streaming to the frontend.
    # Configuration.from_context() still works because configurable
    # is preserved — only the callback chain is cleared.
    try:
        from langchain_core.runnables import RunnableConfig as _RC

        from react_agent.tools import retrieve_documents

        # Strip callbacks from the current config while keeping configurable
        _cfg: _RC = {**config, "callbacks": []}
        _token = var_child_runnable_config.set(_cfg)
        try:
            result = await retrieve_documents(query)
        finally:
            var_child_runnable_config.reset(_token)

        if result and "찾지 못했습니다" not in result and "오류" not in result:
            logger.info("[pre_retrieve] done — context_len=%d", len(result))
            return {"retrieved_context": result}
        logger.info("[pre_retrieve] done — no relevant documents found")
        return {"retrieved_context": ""}
    except Exception:
        logger.warning(
            "[pre_retrieve] failed, proceeding without context", exc_info=True
        )
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
    logger.info("[call_model] start — model=%s", configuration.model)

    # Initialize the model with tool binding. Change the model or add more tools here.
    llm = load_chat_model(configuration.model, temperature=0.1, streaming=True)
    model = llm.bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Prepend conversation summary if available
    logger.info(
        "[call_model] context — summary=%s, retrieved_context=%s",
        f"yes({len(state.summary)} chars)" if state.summary else "no",
        f"yes({len(state.retrieved_context)} chars)"
        if state.retrieved_context
        else "no",
    )
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

    # Strip model_name from response metadata if admin disabled it
    if not configuration.show_model_name:
        response.response_metadata.pop("model_name", None)

    # Log response details
    tool_names = (
        [tc["name"] for tc in response.tool_calls] if response.tool_calls else []
    )
    response_len = len(response.content) if isinstance(response.content, str) else 0
    logger.info(
        "[call_model] done — response_len=%d, tool_calls=%s",
        response_len,
        tool_names if tool_names else "none",
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        logger.warning(
            "[call_model] last step reached — forcing end without tool execution"
        )
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
                    logger.info(
                        "[title_generation] model=%s", configuration.summarization_model
                    )
                    title_llm = load_chat_model(
                        configuration.summarization_model, temperature=0
                    )
                    title_response = await title_llm.ainvoke(
                        [
                            {
                                "role": "user",
                                "content": TITLE_PROMPT.format(
                                    message=human_msgs[0].content
                                ),
                            }
                        ]
                    )
                    raw = title_response.content
                    if isinstance(raw, list):
                        title = " ".join(
                            block["text"]
                            for block in raw
                            if isinstance(block, dict) and block.get("type") == "text"
                        ).strip()[:50]
                    else:
                        title = str(raw).strip()[:50]

                    # Write title to LangGraph thread metadata (single source of truth)
                    try:
                        import httpx

                        langgraph_api_url = os.environ.get(
                            "LANGGRAPH_API_URL", "http://localhost:2024"
                        )
                        langgraph_auth_key = os.environ.get("LANGGRAPH_AUTH_KEY", "")
                        headers: dict[str, str] = {}
                        if langgraph_auth_key:
                            from react_agent.auth import mint_service_jwt

                            service_token = mint_service_jwt(langgraph_auth_key)
                            headers["Authorization"] = f"Bearer {service_token}"

                        if _UUID_RE.match(thread_id):
                            async with httpx.AsyncClient() as http_client:
                                await http_client.patch(
                                    f"{langgraph_api_url}/threads/{thread_id}",
                                    json={"metadata": {"title": title}},
                                    headers=headers,
                                    timeout=5.0,
                                )
                        else:
                            logger.warning(
                                "[title_generation] skipping — thread_id failed UUID validation: %r",
                                thread_id,
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
        logger.info("[route_model_output] → __end__")
        return "__end__"
    # Otherwise we execute the requested actions
    tool_names = [tc["name"] for tc in last_message.tool_calls]
    logger.info("[route_model_output] → tools %s", tool_names)
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
    builder = StateGraph(State, input=InputState, config_schema=Configuration)  # type: ignore[call-arg]

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
