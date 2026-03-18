"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, cast

from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.db import (
    create_thread_for_user,
    create_user,
    get_database_url,
)
from react_agent.prompts import SUMMARIZATION_PROMPT, TITLE_PROMPT
from react_agent.state import InputState, State
from react_agent.tools import TOOLS

logger = logging.getLogger(__name__)

_admin_initialized = False
_db_url: str | None = None


async def ensure_admin_user(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Ensure admin user and current thread exist in the database.

    Creates admin user on first invocation, and registers the current
    thread_id for the admin user on every invocation.

    Args:
        state: The current conversation state.
        config: Configuration for the model run.

    Returns:
        dict: Always returns empty dict (no state changes).
    """
    global _admin_initialized, _db_url  # noqa: PLW0603

    # 1) First call: create admin user (tables must exist beforehand)
    if not _admin_initialized:
        try:
            _db_url = get_database_url()
            await create_user(_db_url, "admin", "Admin")
            _admin_initialized = True
            logger.info("Admin user initialized successfully")
        except Exception:
            logger.exception("Failed to initialize admin user")
            _db_url = None
            _admin_initialized = True
            return {}

    # 2) Every call: register thread for admin user
    if _db_url:
        try:
            configurable = config.get("configurable") or {}
            thread_id = configurable.get("thread_id", "")
            if thread_id:
                await create_thread_for_user(_db_url, "admin", thread_id)
        except Exception:
            logger.exception("Failed to register thread")

    return {}


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
    if _db_url and not response.tool_calls:
        try:
            from langchain_core.messages import HumanMessage

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
                        async with httpx.AsyncClient() as http_client:
                            await http_client.patch(
                                f"{langgraph_api_url}/threads/{thread_id}",
                                json={"metadata": {"title": title}},
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

    builder.add_node(ensure_admin_user)
    builder.add_node(summarize_conversation)
    builder.add_node(call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge("__start__", "ensure_admin_user")
    builder.add_edge("ensure_admin_user", "summarize_conversation")
    builder.add_edge("summarize_conversation", "call_model")

    builder.add_conditional_edges(
        "call_model",
        route_model_output,
    )

    builder.add_edge("tools", "call_model")

    return builder.compile(checkpointer=checkpointer, name="ReAct Agent")


# Module-level graph for langgraph.json backward compatibility
graph = create_graph()
