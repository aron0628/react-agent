"""Tests for user/thread CRUD functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_create_user():
    """create_user should execute INSERT SQL."""
    from react_agent.db import create_user

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "psycopg.AsyncConnection.connect",
        new_callable=AsyncMock,
        return_value=mock_conn,
    ):
        await create_user("postgresql://test", "user1", "Test User")

    mock_conn.execute.assert_called_once()
    call_args = mock_conn.execute.call_args
    assert "INSERT INTO users" in call_args[0][0]
    assert ("user1", "Test User") == call_args[0][1]


@pytest.mark.asyncio
async def test_create_thread_for_user():
    """create_thread_for_user should execute INSERT SQL with FK."""
    from react_agent.db import create_thread_for_user

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "psycopg.AsyncConnection.connect",
        new_callable=AsyncMock,
        return_value=mock_conn,
    ):
        await create_thread_for_user("postgresql://test", "user1", "thread1", "Chat 1")

    mock_conn.execute.assert_called_once()
    call_args = mock_conn.execute.call_args
    assert "INSERT INTO user_threads" in call_args[0][0]


@pytest.mark.asyncio
async def test_get_user_threads():
    """get_user_threads should return list of thread dicts."""
    from react_agent.db import get_user_threads

    mock_cursor = AsyncMock()
    mock_cursor.fetchall.return_value = [
        {
            "thread_id": "t1",
            "title": "Chat 1",
            "created_at": "2026-01-01",
            "updated_at": "2026-01-01",
        },
    ]

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute.return_value = mock_cursor

    with patch(
        "psycopg.AsyncConnection.connect",
        new_callable=AsyncMock,
        return_value=mock_conn,
    ):
        result = await get_user_threads("postgresql://test", "user1")

    assert len(result) == 1
    assert result[0]["thread_id"] == "t1"


@pytest.mark.asyncio
async def test_delete_thread():
    """delete_thread should return True when a row is deleted."""
    from react_agent.db import delete_thread

    mock_cursor = MagicMock()
    mock_cursor.rowcount = 1

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute.return_value = mock_cursor

    with patch(
        "psycopg.AsyncConnection.connect",
        new_callable=AsyncMock,
        return_value=mock_conn,
    ):
        result = await delete_thread("postgresql://test", "user1", "thread1")

    assert result is True


@pytest.mark.asyncio
async def test_delete_thread_not_found():
    """delete_thread should return False when no row is deleted."""
    from react_agent.db import delete_thread

    mock_cursor = MagicMock()
    mock_cursor.rowcount = 0

    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute.return_value = mock_cursor

    with patch(
        "psycopg.AsyncConnection.connect",
        new_callable=AsyncMock,
        return_value=mock_conn,
    ):
        result = await delete_thread("postgresql://test", "user1", "thread1")

    assert result is False
