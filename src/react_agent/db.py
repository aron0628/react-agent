"""Database module for PostgreSQL persistence and user/thread management."""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

_settings_cache: dict[str, str] = {}
_cache_timestamp: float = 0.0
_last_updated_at: str | None = None


async def load_settings_cache(db_url: str) -> None:
    """Query app_settings table and populate the in-memory settings cache.

    Args:
        db_url: PostgreSQL connection URL.
    """
    global _settings_cache, _cache_timestamp, _last_updated_at  # noqa: PLW0603
    import psycopg
    from psycopg.rows import dict_row

    async with await psycopg.AsyncConnection.connect(
        db_url, row_factory=dict_row
    ) as conn:
        cursor = await conn.execute("SELECT key, value, updated_at FROM app_settings")
        rows = await cursor.fetchall()
        _settings_cache = {row["key"]: row["value"] for row in rows}
        # Track the latest updated_at for change detection
        if rows:
            _last_updated_at = str(max(row["updated_at"] for row in rows))
        _cache_timestamp = time.time()


def get_cached_settings() -> dict[str, str]:
    """Return the in-memory settings cache.

    Sync — safe to call from sync classmethods.

    Returns:
        A dict mapping setting keys to values.
    """
    return _settings_cache


def _bump_cache_timestamp() -> None:
    """Update cache timestamp without reloading data."""
    global _cache_timestamp  # noqa: PLW0603
    _cache_timestamp = time.time()


def invalidate_settings_cache() -> None:
    """Force the settings cache to reload on next ensure_settings_loaded call."""
    global _cache_timestamp  # noqa: PLW0603
    _cache_timestamp = 0.0


async def ensure_settings_loaded(db_url: str, ttl: float = 30.0) -> None:
    """Refresh the settings cache if DB settings have changed.

    Uses a two-tier strategy:
    1. Within TTL: skip entirely (no DB hit).
    2. After TTL: lightweight MAX(updated_at) check. Only full-reload if changed.

    Args:
        db_url: PostgreSQL connection URL.
        ttl: Minimum seconds between DB version checks. Defaults to 30.0.
    """
    if time.time() - _cache_timestamp <= ttl:
        return

    import psycopg

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        cursor = await conn.execute(
            "SELECT MAX(updated_at) AS max_updated FROM app_settings"
        )
        row = await cursor.fetchone()
        db_max = str(row[0]) if row and row[0] else None

    if db_max != _last_updated_at:
        await load_settings_cache(db_url)
    else:
        # DB unchanged — just bump the timestamp to avoid re-checking within TTL
        _bump_cache_timestamp()


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


async def get_user_role(db_url: str, user_id: str) -> str:
    """user_id로 users 테이블에서 role 조회. fail-closed: DB 장애 시 'user' 반환.

    Args:
        db_url: PostgreSQL connection URL.
        user_id: The user whose role to look up.

    Returns:
        The user's role string ('admin', 'user', etc.).
        Returns 'admin' if user_id is empty (backward compat).
        Returns 'user' on DB failure or unknown user (fail-closed).
    """
    if not user_id:
        return "admin"  # user_id 없으면 기존 동작 유지 (전체 검색)

    try:
        import psycopg

        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT role FROM users WHERE user_id = %s AND is_active = true",
                    (user_id,),
                )
                row = await cur.fetchone()
                if row:
                    return row[0]
                return "user"  # 사용자를 못 찾으면 최소 권한
    except Exception:
        logger.warning(
            "Failed to lookup role for user_id=%s, defaulting to 'user'",
            user_id,
        )
        return "user"  # fail-closed


async def create_user(
    db_url: str,
    user_id: str,
    display_name: str = "",
    email: str = "",
) -> None:
    """Create a new user.

    Args:
        db_url: PostgreSQL connection URL.
        user_id: Unique identifier for the user.
        display_name: Optional display name for the user.
        email: Email address for the user.
    """
    import psycopg

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        await conn.execute(
            "INSERT INTO users (user_id, display_name, email) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id, display_name, email),
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


async def get_user_threads(db_url: str, user_id: str) -> list[dict[str, Any]]:
    """Get all conversation threads for a user.

    Args:
        db_url: PostgreSQL connection URL.
        user_id: The user whose threads to retrieve.

    Returns:
        A list of thread dictionaries with thread_id, title, created_at, updated_at.
    """
    import psycopg
    from psycopg.rows import dict_row

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
    """Create a checkpointer for langgraph.json config.

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
