"""LangGraph API key authentication handler."""

import hmac
import logging
import os

from langgraph_sdk import Auth

logger = logging.getLogger(__name__)

auth = Auth()


@auth.authenticate
async def authenticate(authorization: str | None) -> str:
    """Validate API key from Authorization header."""
    expected_key = os.environ.get("LANGGRAPH_AUTH_KEY", "")
    if not expected_key:
        logger.warning("Auth rejected: LANGGRAPH_AUTH_KEY not configured")
        raise Auth.exceptions.HTTPException(
            status_code=500, detail="Auth not configured"
        )

    if not authorization:
        logger.warning("Auth rejected: missing Authorization header")
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing authorization"
        )

    if not authorization.startswith("Bearer "):
        logger.warning(
            "Auth rejected: invalid authorization scheme (expected 'Bearer ')"
        )
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid authorization scheme"
        )

    token = authorization[7:]  # len("Bearer ") == 7

    if not hmac.compare_digest(token, expected_key):
        logger.warning("Auth rejected: invalid API key")
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid API key"
        )

    return "service-user"
