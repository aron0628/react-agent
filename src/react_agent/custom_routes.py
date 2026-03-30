"""Custom HTTP routes mounted on the LangGraph server."""

import logging

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from react_agent.db import invalidate_settings_cache

logger = logging.getLogger(__name__)


async def invalidate_cache(request: Request) -> JSONResponse:
    """Invalidate the in-memory settings cache.

    Called by the admin panel after saving settings so that the next
    graph invocation picks up the latest values from the database.
    """
    invalidate_settings_cache()
    logger.info("[custom_routes] settings cache invalidated via API")
    return JSONResponse({"status": "ok", "message": "Settings cache invalidated"})


app = Starlette(
    routes=[
        Route("/invalidate-settings-cache", invalidate_cache, methods=["POST"]),
    ]
)
