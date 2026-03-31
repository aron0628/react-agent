"""LangGraph API JWT authentication handler."""

import datetime
import hmac
import logging
import os

import jwt
from langgraph_sdk import Auth

logger = logging.getLogger(__name__)

auth = Auth()


def mint_service_jwt(secret: str, ttl_seconds: int = 60) -> str:
    """Mint a short-lived JWT for internal service-to-service calls.

    Args:
        secret: The HMAC-SHA256 signing key.
        ttl_seconds: Token time-to-live in seconds. Defaults to 60.

    Returns:
        An encoded JWT string.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    payload = {
        "sub": "service-internal",
        "iss": "agent-chat-ui",
        "aud": "react-agent",
        "iat": now,
        "exp": now + datetime.timedelta(seconds=ttl_seconds),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


@auth.authenticate
async def authenticate(authorization: str | None) -> str:
    """Validate JWT from Authorization header and extract user identity.

    Args:
        authorization: The raw Authorization header value.

    Returns:
        The user_id extracted from the JWT ``sub`` claim.

    Raises:
        Auth.exceptions.HTTPException: On missing, invalid, or expired tokens.
    """
    jwt_secret = os.environ.get("LANGGRAPH_AUTH_KEY", "")
    if not jwt_secret:
        logger.critical(
            "LANGGRAPH_AUTH_KEY is not configured — all requests will be rejected"
        )
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Service unavailable"
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

    try:
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            options={"require": ["sub", "exp", "iss", "aud"]},
            issuer="agent-chat-ui",
            audience="react-agent",
            leeway=10,
        )
    except jwt.ExpiredSignatureError:
        logger.warning("Auth rejected: token expired")
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Token expired"
        )
    except jwt.InvalidTokenError as exc:
        logger.warning("Auth rejected: invalid token — %s", exc)
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid token"
        )

    user_id = payload.get("sub")
    if not user_id:
        logger.warning("Auth rejected: missing sub claim")
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing user identity"
        )

    return user_id
