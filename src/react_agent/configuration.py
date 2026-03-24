"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.config import get_config

from react_agent import prompts


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
        default="gpt-4.1",
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
            default="gpt-4.1-mini",
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
        metadata={
            "description": "Maximum number of documents to retrieve."
        },
    )

    rag_max_rewrite_attempts: int = field(
        default=1,
        metadata={
            "description": "Maximum query rewrite attempts when no relevant docs found."
        },
    )

    rag_grading_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = (
        field(
            default="gpt-4.1-mini",
            metadata={
                "description": "Model for batch document relevance grading."
            },
        )
    )

    rag_max_response_tokens: int = field(
        default=4000,
        metadata={
            "description": "Token budget cap for retrieve_documents tool response."
        },
    )

    enable_raptor: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_RAPTOR", "false").lower()
        == "true",
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
        metadata={
            "description": "Max RAPTOR summary clusters to retrieve in Stage 1."
        },
    )

    enable_hybrid_search: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_HYBRID_SEARCH", "false").lower()
        == "true",
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
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from the current context."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
