"""Tests for database module."""

import os
from unittest.mock import patch

import pytest

from react_agent.db import get_database_url


def test_get_database_url_from_env():
    """get_database_url should build URL from individual env vars."""
    env = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "testdb",
        "DB_USER": "testuser",
        "DB_PASSWORD": "testpass",
    }
    with patch.dict(os.environ, env, clear=False):
        url = get_database_url()
    assert url == "postgresql://testuser:testpass@localhost:5432/testdb"


def test_get_database_url_special_chars():
    """get_database_url should encode special characters in password."""
    env = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "testdb",
        "DB_USER": "testuser",
        "DB_PASSWORD": "p@ss#word!",
    }
    with patch.dict(os.environ, env, clear=False):
        url = get_database_url()
    assert "p%40ss%23word%21" in url


def test_get_database_url_missing_host_raises():
    """get_database_url should raise RuntimeError when DB_HOST is missing."""
    env_without_host = {k: v for k, v in os.environ.items() if k != "DB_HOST"}
    with patch.dict(os.environ, env_without_host, clear=True):
        with pytest.raises(RuntimeError, match="DB_HOST"):
            get_database_url()


def test_get_database_url_defaults():
    """get_database_url should use defaults for optional fields."""
    with patch.dict(os.environ, {"DB_HOST": "myhost"}, clear=True):
        url = get_database_url()
    assert "myhost" in url
    assert "5432" in url
    assert "app_db" in url
    assert "app_user" in url
