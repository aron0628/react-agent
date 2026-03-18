"""Database module for PostgreSQL persistence and user/thread management."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote_plus


def get_database_url() -> str:
    """Build PostgreSQL connection URL from individual environment variables.

    Reads DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD from environment.
    Uses quote_plus() to safely encode passwords with special characters.

    Returns:
        A PostgreSQL connection URL string.

    Raises:
        RuntimeError: If DB_HOST is not set in environment variables.
    """
    host = os.environ.get("DB_HOST")
    if not host:
        raise RuntimeError(
            "DB_HOST environment variable is not set. "
            "Set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD for database access."
        )
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "app_db")
    user = os.environ.get("DB_USER", "app_user")
    password = quote_plus(os.environ.get("DB_PASSWORD", ""))
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


async def setup_user_tables(db_url: str) -> None:
    """Create users and user_threads tables if they don't exist.

    Args:
        db_url: PostgreSQL connection URL.
    """
    import psycopg  # type: ignore[import-not-found]

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        sql_path = os.path.join(
            os.path.dirname(__file__), "migrations", "001_user_tables.sql"
        )
        with open(sql_path) as f:
            sql = f.read()
        await conn.execute(sql)
        await conn.commit()


async def create_user(db_url: str, user_id: str, display_name: str = "") -> None:
    """Create a new user.

    Args:
        db_url: PostgreSQL connection URL.
        user_id: Unique identifier for the user.
        display_name: Optional display name for the user.
    """
    import psycopg

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        await conn.execute(
            "INSERT INTO users (user_id, display_name) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id, display_name),
        )
        await conn.commit()


async def create_thread_for_user(
    db_url: str, user_id: str, thread_id: str, title: str = ""
) -> None:
    """Create a new conversation thread for a user.

    Args:
        db_url: PostgreSQL connection URL.
        user_id: The user who owns the thread.
        thread_id: Unique identifier for the thread.
        title: Optional title for the conversation thread.
    """
    import psycopg

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        await conn.execute(
            "INSERT INTO user_threads (thread_id, user_id, title) VALUES (%s, %s, %s) "
            "ON CONFLICT (thread_id) DO NOTHING",
            (thread_id, user_id, title),
        )
        await conn.commit()


async def update_thread_title(db_url: str, thread_id: str, title: str) -> None:
    """Update thread title only if currently empty.

    Args:
        db_url: PostgreSQL connection URL.
        thread_id: The thread to update.
        title: New title for the thread.
    """
    import psycopg

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        await conn.execute(
            "UPDATE user_threads SET title = %s, updated_at = NOW() "
            "WHERE thread_id = %s AND title = ''",
            (title, thread_id),
        )
        await conn.commit()


async def get_user_threads(db_url: str, user_id: str) -> list[dict[str, Any]]:
    """Get all conversation threads for a user.

    Args:
        db_url: PostgreSQL connection URL.
        user_id: The user whose threads to retrieve.

    Returns:
        A list of thread dictionaries with thread_id, title, created_at, updated_at.
    """
    import psycopg
    from psycopg.rows import dict_row  # type: ignore[import-not-found]

    async with await psycopg.AsyncConnection.connect(
        db_url, row_factory=dict_row
    ) as conn:
        cursor = await conn.execute(
            "SELECT thread_id, title, created_at, updated_at "
            "FROM user_threads WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def delete_thread(db_url: str, user_id: str, thread_id: str) -> bool:
    """Delete a conversation thread.

    Args:
        db_url: PostgreSQL connection URL.
        user_id: The user who owns the thread.
        thread_id: The thread to delete.

    Returns:
        True if a thread was deleted, False if not found.
    """
    import psycopg

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        cursor = await conn.execute(
            "DELETE FROM user_threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.commit()
        return bool(cursor.rowcount > 0)


def create_checkpointer() -> Any:
    """Factory for langgraph.json checkpointer config.

    Called by langgraph_api's _yield_checkpointer as a callable.
    Returns an async context manager that the adapter enters via
    ``async with value as ctx_value``.

    Returns:
        AsyncPostgresSaver async context manager (from_conn_string).

    Raises:
        RuntimeError: If DB_HOST is not set (via get_database_url()).
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    db_uri = get_database_url()
    return AsyncPostgresSaver.from_conn_string(db_uri)


async def run_with_persistence(
    config: dict[str, Any] | None = None,
) -> None:
    """Run the graph with PostgreSQL persistence.

    Demonstrates the async context manager lifecycle pattern for
    AsyncPostgresSaver. The checkpointer connection stays alive
    for the entire scope.

    Args:
        config: Optional configuration dict for graph invocation.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from react_agent.graph import create_graph

    db_uri = get_database_url()
    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        graph = create_graph(checkpointer=checkpointer)
        if config:
            await graph.ainvoke(config)


async def create_checkpointer_from_pool(
    pool: Any,
) -> Any:
    """Create an AsyncPostgresSaver from an existing connection pool.

    Use this pattern for long-lived processes where the connection pool
    outlives individual graph invocations.

    Use this pattern for long-lived processes where the connection pool
    outlives individual graph invocations.

    Args:
        pool: An AsyncConnectionPool instance from psycopg_pool.

    Returns:
        A configured AsyncPostgresSaver instance with tables initialized.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    checkpointer = AsyncPostgresSaver(pool)
    return checkpointer
