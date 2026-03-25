from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryCaptionType,
    QueryAnswerType,
    SemanticErrorMode,
    VectorizedQuery,
)

from embed import ChunkEmbedder
from index import INDEX_NAME_DEFAULT, SEMANTIC_CONFIG_NAME

logger = logging.getLogger(__name__)

SearchMode = Literal["keyword", "vector", "hybrid", "semantic"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single ranked result returned by the search pipeline."""

    rank: int
    chunk_id: str
    score: float                    # BM25 / RRF / reranker score (depends on mode)
    reranker_score: Optional[float] # Populated only in semantic mode
    content: str                    # Chunk text
    caption: Optional[str]          # Semantic caption (semantic mode only)
    caption_highlights: Optional[str]
    doc_id: str
    blob_name: str
    source_url: str
    file_type: str
    category: str
    heading: Optional[str]
    page_hint: Optional[int]
    chunk_index: int
    token_count: int

    def __str__(self) -> str:
        score_info = f"score={self.score:.4f}"
        if self.reranker_score is not None:
            score_info += f"  reranker={self.reranker_score:.4f}"
        source = self.blob_name or self.doc_id
        heading = f"  [{self.heading}]" if self.heading else ""
        page = f"  p.{self.page_hint}" if self.page_hint else ""
        cap = f"\n  📌 Caption: {self.caption}" if self.caption else ""
        return (
            f"[{self.rank}] {source}{heading}{page}  {score_info}\n"
            f"  {self.content[:300]}{'…' if len(self.content) > 300 else ''}"
            f"{cap}"
        )


# ---------------------------------------------------------------------------
# Default fields to retrieve
# ---------------------------------------------------------------------------

_DEFAULT_SELECT = [
    "id",
    "content",
    "doc_id",
    "blob_name",
    "source_url",
    "file_type",
    "category",
    "heading",
    "page_hint",
    "chunk_index",
    "token_count",
]


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class KnowledgeBaseSearcher:
    """
    Executes hybrid / semantic search against an Azure AI Search index.

    Parameters
    ----------
    search_endpoint : str
        Azure AI Search service endpoint.
    query_key : str
        Query API key (read-only; use admin key if you need write access).
    openai_endpoint : str
        Azure OpenAI endpoint (for query-time embedding).
    openai_key : str
        Azure OpenAI API key.
    embedding_deployment : str
        Deployment name for the embedding model (must match what was used at index time).
    index_name : str
        Name of the Azure AI Search index.
    """

    def __init__(
        self,
        search_endpoint: str,
        query_key: str,
        openai_endpoint: str,
        openai_key: str,
        embedding_deployment: str = "text-embedding-ada-002",
        index_name: str = INDEX_NAME_DEFAULT,
    ) -> None:
        self._search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(query_key),
        )
        self._embedder = ChunkEmbedder(
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
            deployment_name=embedding_deployment,
            normalise=True,
        )

    # ------------------------------------------------------------------
    # Public search entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        top_n: int = 5,
        filter: Optional[str] = None,
        select: Optional[list[str]] = None,
        min_reranker_score: Optional[float] = None,
    ) -> list[SearchResult]:
        """
        Execute a search and return top-n ranked results.

        Parameters
        ----------
        query : str
            Natural-language query string.
        mode : "keyword" | "vector" | "hybrid" | "semantic"
            Search strategy:
            - keyword  → BM25 only (fast, lexical)
            - vector   → ANN vector only (semantic similarity, no keywords)
            - hybrid   → BM25 + vector fused with RRF (best for most use cases)
            - semantic → hybrid + semantic re-ranker + captions (most accurate)
        top_n : int
            Number of results to return (default 5).
        filter : str | None
            OData filter, e.g. ``"category eq 'manuals' and file_type eq 'pdf'"``
        select : list[str] | None
            Fields to return.  Defaults to _DEFAULT_SELECT.
        min_reranker_score : float | None
            If set and mode="semantic", drop results below this threshold (0–4 scale).
        """
        select = select or _DEFAULT_SELECT
        logger.info("search(query=%r, mode=%s, top_n=%d)", query, mode, top_n)

        # Build query embedding for vector/hybrid/semantic modes
        query_vector = None
        if mode in ("vector", "hybrid", "semantic"):
            query_vector = self._embedder.embed_text(query)

        # Dispatch
        if mode == "keyword":
            raw_results = self._keyword_search(query, top_n, filter, select)
        elif mode == "vector":
            raw_results = self._vector_search(query_vector, top_n, filter, select)
        elif mode == "hybrid":
            raw_results = self._hybrid_search(query, query_vector, top_n, filter, select)
        elif mode == "semantic":
            raw_results = self._semantic_search(query, query_vector, top_n, filter, select)
        else:
            raise ValueError(f"Unknown search mode: {mode!r}")

        results = [self._to_result(r, rank=i + 1) for i, r in enumerate(raw_results)]

        # Optional reranker score filter (semantic mode)
        if min_reranker_score is not None:
            results = [r for r in results if (r.reranker_score or 0) >= min_reranker_score]

        logger.info("Returning %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal search strategies
    # ------------------------------------------------------------------

    def _keyword_search(self, query, top_n, filter, select):
        return list(
            self._search_client.search(
                search_text=query,
                top=top_n,
                filter=filter,
                select=select,
                query_type="full",       # Lucene full syntax for phrase/wildcard
            )
        )

    def _vector_search(self, query_vector, top_n, filter, select):
        vec_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_n,
            fields="embedding",
        )
        return list(
            self._search_client.search(
                search_text=None,
                vector_queries=[vec_query],
                top=top_n,
                filter=filter,
                select=select,
            )
        )

    def _hybrid_search(self, query, query_vector, top_n, filter, select):
        vec_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_n,
            fields="embedding",
        )
        return list(
            self._search_client.search(
                search_text=query,
                vector_queries=[vec_query],
                top=top_n,
                filter=filter,
                select=select,
                # RRF fusion is automatic when both search_text and vector_queries are provided
            )
        )

    def _semantic_search(self, query, query_vector, top_n, filter, select):
        """
        Hybrid search with semantic re-ranking and caption generation.
        Requests up to 3 extractive answers from the top results.
        """
        vec_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_n * 2,   # Over-fetch for re-ranker to pick from
            fields="embedding",
        )
        return list(
            self._search_client.search(
                search_text=query,
                vector_queries=[vec_query],
                top=top_n,
                filter=filter,
                select=select,
                query_type="semantic",
                semantic_configuration_name=SEMANTIC_CONFIG_NAME,
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                query_answer_count=3,
                semantic_error_mode=SemanticErrorMode.PARTIAL,  # Graceful fallback
            )
        )

    # ------------------------------------------------------------------
    # Result deserialiser
    # ------------------------------------------------------------------

    @staticmethod
    def _to_result(raw: dict, rank: int) -> SearchResult:
        """Convert a raw Azure Search result dict to a SearchResult."""
        caption_obj = getattr(raw, "@search.captions", None)
        caption_text = None
        caption_highlights = None
        if caption_obj:
            first = caption_obj[0] if caption_obj else None
            if first:
                caption_text = getattr(first, "text", None)
                caption_highlights = getattr(first, "highlights", None)

        return SearchResult(
            rank=rank,
            chunk_id=raw.get("id", ""),
            score=raw.get("@search.score", 0.0),
            reranker_score=raw.get("@search.reranker_score"),
            content=raw.get("content", ""),
            caption=caption_text,
            caption_highlights=caption_highlights,
            doc_id=raw.get("doc_id", ""),
            blob_name=raw.get("blob_name", ""),
            source_url=raw.get("source_url", ""),
            file_type=raw.get("file_type", ""),
            category=raw.get("category", ""),
            heading=raw.get("heading") or None,
            page_hint=raw.get("page_hint") or None,
            chunk_index=raw.get("chunk_index", 0),
            token_count=raw.get("token_count", 0),
        )

    # ------------------------------------------------------------------
    # Convenience: compare modes side-by-side
    # ------------------------------------------------------------------

    def compare_modes(
        self,
        query: str,
        top_n: int = 3,
        filter: Optional[str] = None,
    ) -> dict[SearchMode, list[SearchResult]]:
        """Run the same query under all four modes and return results by mode."""
        return {
            mode: self.search(query, mode=mode, top_n=top_n, filter=filter)
            for mode in ("keyword", "vector", "hybrid", "semantic")
        }