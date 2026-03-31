"""Define the configurable parameters for the agent."""

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.config import get_config

from react_agent import prompts

# DB key → Configuration field name aliases
_KEY_ALIASES: dict[str, str] = {
    "summary_message_threshold": "summarization_threshold",
}

# Fields that callers must not override via configurable dict
_CALLER_BLOCKLIST: set[str] = {"system_prompt", "model", "show_model_name", "user_role"}

# Allowlist of valid model identifiers (with and without provider prefix)
_ALLOWED_MODELS: set[str] = {
    # OpenAI
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/o1",
    "openai/o1-mini",
    "openai/o3-mini",
    # Anthropic
    "anthropic/claude-opus-4-5",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-3-5",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-haiku-20241022",
    # Google
    "google_genai/gemini-3.1-pro-preview",
    "google_genai/gemini-3.1-flash-lite-preview",
    "google_genai/gemini-3-flash-preview",
    # xAI
    "xai/grok-4.20-0309-reasoning",
    "xai/grok-4.20-0309-non-reasoning",
    "xai/grok-4.20-multi-agent-0309",
    "xai/grok-4-1-fast-reasoning",
    "xai/grok-4-1-fast-non-reasoning",
    # OpenAI (without provider prefix)
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "o1",
    "o1-mini",
    "o3-mini",
}

# Maximum allowed length for system_prompt loaded from DB
_MAX_SYSTEM_PROMPT_LENGTH: int = 10000


def _apply_key_aliases(settings: dict[str, str]) -> dict[str, str]:
    """Remap DB setting keys to Configuration field names.

    Args:
        settings: Raw key-value settings from the database.

    Returns:
        A new dict with aliased keys remapped to their canonical field names.
    """
    return {_KEY_ALIASES.get(k, k): v for k, v in settings.items()}


def _coerce_field_types(merged: dict[str, Any], cls: type) -> dict[str, Any]:
    """Coerce string values from DB to match dataclass field types.

    Args:
        merged: Merged settings dict (may contain string values from DB).
        cls: The dataclass class to inspect for field defaults.

    Returns:
        A new dict with values coerced to the correct Python types.
    """
    type_map: dict[str, type] = {}
    for f in fields(cls):
        if f.default is not MISSING:
            type_map[f.name] = type(f.default)
        elif f.default_factory is not MISSING:
            type_map[f.name] = type(f.default_factory())

    result: dict[str, Any] = {}
    for k, v in merged.items():
        if isinstance(v, str) and k in type_map:
            target = type_map[k]
            try:
                if target is int:
                    result[k] = int(v)
                elif target is float:
                    result[k] = float(v)
                elif target is bool:
                    result[k] = v.lower() in ("true", "1", "yes")
                else:
                    result[k] = v
            except (ValueError, TypeError):
                result[k] = v
        else:
            result[k] = v
    return result


logger = logging.getLogger(__name__)

# Fields that hold LLM model identifiers
_MODEL_FIELDS = {"model", "summarization_model", "rag_grading_model"}


def _ensure_provider_prefix(model_name: str) -> str:
    """Auto-prefix 'openai/' if no provider prefix exists.

    Args:
        model_name: A model name string, optionally with a provider prefix.

    Returns:
        The model name with a provider prefix (e.g. 'openai/gpt-4.1').
    """
    if "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def _is_valid_model_name(name: str) -> bool:
    """Check if a string looks like a valid model name (not a pure number).

    Args:
        name: The model name to validate.

    Returns:
        True if the name looks like a valid model identifier.
    """
    stripped = name.strip()
    if not stripped:
        return False
    try:
        float(stripped)
        return False
    except ValueError:
        return True


def _validate_db_settings(merged: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize DB-sourced settings before applying to Configuration.

    Args:
        merged: Settings dict loaded from the database (post alias remapping).

    Returns:
        A sanitized copy with invalid model names and oversized prompts removed.
    """
    result = dict(merged)

    # Validate model fields against the allowlist
    for fname in _MODEL_FIELDS:
        val = result.get(fname)
        if isinstance(val, str):
            normalized = val.strip()
            if normalized not in _ALLOWED_MODELS:
                logger.warning(
                    "[config] DB model value %r for field %r not in allowlist, "
                    "discarding (will use default)",
                    val,
                    fname,
                )
                del result[fname]

    # Enforce system_prompt length limit
    prompt_val = result.get("system_prompt")
    if isinstance(prompt_val, str) and len(prompt_val) > _MAX_SYSTEM_PROMPT_LENGTH:
        logger.warning(
            "[config] DB system_prompt length %d exceeds max %d, "
            "discarding (will use default)",
            len(prompt_val),
            _MAX_SYSTEM_PROMPT_LENGTH,
        )
        del result["system_prompt"]

    return result


def _validate_model_fields(instance: Configuration) -> None:
    """Validate and fix model fields, falling back to defaults for invalid values.

    Args:
        instance: A Configuration instance to validate in-place.
    """
    defaults = {f.name: f.default for f in fields(instance) if f.name in _MODEL_FIELDS}
    for fname in _MODEL_FIELDS:
        val = getattr(instance, fname, None)
        if isinstance(val, str):
            if not _is_valid_model_name(val):
                default_val = defaults.get(fname, "openai/gpt-4.1-mini")
                logger.warning(
                    "[config] invalid model name %r for field %r, "
                    "falling back to default %r",
                    val,
                    fname,
                    default_val,
                )
                val = str(default_val)
            setattr(instance, fname, _ensure_provider_prefix(val))


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4.1-mini",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    summarization_threshold: int = field(
        default=20,
        metadata={
            "description": "Number of messages before triggering conversation summarization."
        },
    )

    summarization_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = (
        field(
            default="openai/gpt-4.1-mini",
            metadata={"description": "Model to use for conversation summarization."},
        )
    )

    embedding_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="solar-embedding-1-large",
        metadata={
            "description": "Upstage embedding model. "
            "Auto-appends -query/-passage suffix."
        },
    )

    embedding_dimensions: int = field(
        default=4096,
        metadata={
            "description": "Embedding vector dimensions. "
            "Must match the DB column (solar-embedding-1-large = 4096)."
        },
    )

    rag_max_distance: float = field(
        default=0.5,
        metadata={
            "description": "Max cosine distance for vector search "
            "(pgvector <=>: 0=identical, 2=opposite). Lower is more similar."
        },
    )

    rag_max_results: int = field(
        default=5,
        metadata={"description": "Maximum number of documents to retrieve."},
    )

    rag_max_rewrite_attempts: int = field(
        default=1,
        metadata={
            "description": "Maximum query rewrite attempts when no relevant docs found."
        },
    )

    rag_grading_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = (
        field(
            default="openai/gpt-4.1-mini",
            metadata={"description": "Model for batch document relevance grading."},
        )
    )

    rag_max_response_tokens: int = field(
        default=4000,
        metadata={
            "description": "Token budget cap for retrieve_documents tool response."
        },
    )

    enable_raptor: bool = field(
        default_factory=lambda: (
            os.environ.get("ENABLE_RAPTOR", "false").lower() == "true"
        ),
        metadata={
            "description": "Enable 2-stage RAPTOR retrieval. "
            "When enabled, searches raptor_summaries first, "
            "then fetches leaf chunks by cluster indices. "
            "Can be set via ENABLE_RAPTOR env var."
        },
    )

    raptor_max_distance: float = field(
        default=0.8,
        metadata={
            "description": "Max cosine distance for RAPTOR summary search "
            "(higher than leaf threshold since summaries are more abstract)."
        },
    )

    raptor_top_k: int = field(
        default=5,
        metadata={"description": "Max RAPTOR summary clusters to retrieve in Stage 1."},
    )

    show_model_name: bool = field(
        default=True,
        metadata={"description": "채팅 응답에 사용된 모델명 표시 여부."},
    )

    user_id: str = field(
        default="",
        metadata={
            "description": "현재 사용자 ID. 빈 문자열이면 전체 문서 검색 (admin 동작).",
        },
    )

    enable_hybrid_search: bool = field(
        default_factory=lambda: (
            os.environ.get("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
        ),
        metadata={
            "description": "하이브리드 검색 활성화 (BM25 + Dense). env: ENABLE_HYBRID_SEARCH",
            "env": "ENABLE_HYBRID_SEARCH",
        },
    )

    hybrid_alpha: float = field(
        default_factory=lambda: float(os.environ.get("HYBRID_ALPHA", "0.7")),
        metadata={
            "description": "Dense 가중치 (0.0=순수 Sparse, 1.0=순수 Dense). env: HYBRID_ALPHA",
            "env": "HYBRID_ALPHA",
        },
    )

    bm25_top_k: int = field(
        default_factory=lambda: int(os.environ.get("BM25_TOP_K", "20")),
        metadata={
            "description": "BM25 키워드 검색 후보 수. env: BM25_TOP_K",
            "env": "BM25_TOP_K",
        },
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        from react_agent.db import get_cached_settings

        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        # Start with cached DB settings (validated), then override with configurable values
        cached = _apply_key_aliases(get_cached_settings())
        db_fields: dict[str, Any] = {k: v for k, v in cached.items() if k in _fields}
        db_fields = _validate_db_settings(db_fields)
        merged: dict[str, Any] = dict(db_fields)
        merged.update(
            {
                k: v
                for k, v in configurable.items()
                if k in _fields and k not in _CALLER_BLOCKLIST
            }
        )
        merged = _coerce_field_types(merged, cls)
        instance = cls(**merged)
        _validate_model_fields(instance)
        return instance

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from the current context."""
        from react_agent.db import get_cached_settings

        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        # Start with cached DB settings (validated), then override with configurable values
        cached = _apply_key_aliases(get_cached_settings())
        db_fields: dict[str, Any] = {k: v for k, v in cached.items() if k in _fields}
        db_fields = _validate_db_settings(db_fields)
        merged: dict[str, Any] = dict(db_fields)
        merged.update(
            {
                k: v
                for k, v in configurable.items()
                if k in _fields and k not in _CALLER_BLOCKLIST
            }
        )
        merged = _coerce_field_types(merged, cls)
        instance = cls(**merged)
        _validate_model_fields(instance)
        return instance
