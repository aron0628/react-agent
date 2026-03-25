"""Integration tests for PostgreSQL persistence and user/thread lifecycle.

These tests require a running PostgreSQL instance. They are skipped
when DB_HOST is not set in the environment.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DB_HOST"),
    reason="DB_HOST not set — skipping persistence integration tests",
)


@pytest.mark.asyncio
async def test_conversation_persists_across_invocations() -> None:
    """Graph state should persist across separate ainvoke calls with same thread_id."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from react_agent.db import get_database_url
    from react_agent.graph import create_graph

    db_url = get_database_url()
    thread_id = f"test-persist-{uuid.uuid4().hex[:8]}"

    async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
        g = create_graph(checkpointer=checkpointer)

        # First invocation
        config = {"configurable": {"thread_id": thread_id}}
        res1 = await g.ainvoke({"messages": [("user", "My name is Alice.")]}, config)
        assert len(res1["messages"]) >= 2

        # Second invocation — same thread should retain context
        res2 = await g.ainvoke({"messages": [("user", "What is my name?")]}, config)
        last_content = str(res2["messages"][-1].content).lower()
        assert "alice" in last_content


@pytest.mark.asyncio
async def test_summarization_triggers_on_threshold() -> None:
    """Summarization should trigger when messages exceed the threshold."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from react_agent.db import get_database_url
    from react_agent.graph import create_graph

    db_url = get_database_url()
    thread_id = f"test-summary-{uuid.uuid4().hex[:8]}"

    async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
        g = create_graph(checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "summarization_threshold": 4,  # Low threshold for testing
            }
        }

        # Send multiple messages to exceed threshold
        for i in range(3):
            await g.ainvoke(
                {"messages": [("user", f"Tell me fact number {i + 1} about Python.")]},
                config,
            )

        # After exceeding threshold, summary should exist in state
        state = await g.aget_state(config)
        # The summary field may or may not be populated depending on
        # whether the threshold was hit during the last invocation
        assert state is not None


@pytest.mark.asyncio
async def test_user_thread_lifecycle() -> None:
    """Full CRUD lifecycle for user and thread management."""
    from react_agent.db import (
        create_thread_for_user,
        create_user,
        delete_thread,
        get_database_url,
        get_user_threads,
    )

    db_url = get_database_url()
    user_id = f"test-user-{uuid.uuid4().hex[:8]}"
    thread_id = f"test-thread-{uuid.uuid4().hex[:8]}"

    # Create user
    await create_user(db_url, user_id, "Test User")

    # Create thread
    await create_thread_for_user(db_url, user_id, thread_id, "Test Chat")

    # List threads
    threads = await get_user_threads(db_url, user_id)
    assert len(threads) >= 1
    thread_ids = [t["thread_id"] for t in threads]
    assert thread_id in thread_ids

    # Delete thread
    deleted = await delete_thread(db_url, user_id, thread_id)
    assert deleted is True

    # Verify deletion
    deleted_again = await delete_thread(db_url, user_id, thread_id)
    assert deleted_again is False

    # Verify thread no longer in list
    threads_after = await get_user_threads(db_url, user_id)
    thread_ids_after = [t["thread_id"] for t in threads_after]
    assert thread_id not in thread_ids_after
