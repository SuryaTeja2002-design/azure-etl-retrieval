from __future__ import annotations

import json
import logging
from typing import Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from embed import EmbeddedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_NAME_DEFAULT = "knowledge-base-index"
SEMANTIC_CONFIG_NAME = "semantic-config"
VECTOR_PROFILE_NAME = "vector-profile"
HNSW_CONFIG_NAME = "hnsw-config"


# ---------------------------------------------------------------------------
# Index schema builder
# ---------------------------------------------------------------------------

def _build_index_schema(index_name: str, embedding_dim: int) -> SearchIndex:
    """Construct the SearchIndex definition with vector + semantic config."""

    fields = [
        # Key
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),

        # Main searchable content (BM25 + semantic)
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),

        # Dense vector – must match embedding model dimension
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),

        # Metadata fields
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="blob_name", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="source_url", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="file_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="heading", type=SearchFieldDataType.String),
        SimpleField(name="page_hint", type=SearchFieldDataType.Int32, filterable=True, retrievable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, retrievable=True),
        SimpleField(name="token_count", type=SearchFieldDataType.Int32, retrievable=True),
    ]

    # HNSW approximate nearest-neighbour algorithm
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE_NAME, algorithm_configuration_name=HNSW_CONFIG_NAME)],
        algorithms=[
            HnswAlgorithmConfiguration(
                name=HNSW_CONFIG_NAME,
                parameters={
                    "m": 4,                # Connections per node
                    "efConstruction": 400, # Build-time quality (higher = slower build, better recall)
                    "efSearch": 500,       # Query-time quality
                    "metric": "cosine",
                },
            )
        ],
    )

    # Semantic ranking configuration
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="heading")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


# ---------------------------------------------------------------------------
# Document serialiser
# ---------------------------------------------------------------------------

def _chunk_to_document(ec: EmbeddedChunk) -> dict[str, Any]:
    """Convert an EmbeddedChunk into the flat dict expected by the Search SDK."""
    meta = ec.metadata
    return {
        "id": ec.chunk_id,
        "content": ec.text,
        "embedding": ec.embedding,
        "doc_id": ec.chunk.doc_id,
        "blob_name": meta.get("blob_name", ""),
        "source_url": meta.get("source_url", ""),
        "file_type": meta.get("file_type", ""),
        "category": meta.get("category", ""),
        "heading": meta.get("heading") or "",
        "page_hint": meta.get("page_hint") or 0,
        "chunk_index": meta.get("chunk_index", 0),
        "token_count": meta.get("token_count", 0),
    }


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class KnowledgeBaseIndexer:
    """
    Creates (or updates) an Azure AI Search index and uploads embedded chunks.

    Parameters
    ----------
    search_endpoint : str
        Azure AI Search service endpoint, e.g.
        "https://my-search.search.windows.net"
    admin_key : str
        Azure AI Search admin API key (needed for index management).
    index_name : str
        Name of the target index (created if it does not exist).
    batch_size : int
        Documents per upload batch (max 1000 per Azure limits).
    """

    def __init__(
        self,
        search_endpoint: str,
        admin_key: str,
        index_name: str = INDEX_NAME_DEFAULT,
        batch_size: int = 500,
    ) -> None:
        credential = AzureKeyCredential(admin_key)
        self._index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
        self._search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=credential,
        )
        self.index_name = index_name
        self.batch_size = batch_size

    # ------------------------------------------------------------------

    def create_or_update_index(self, embedding_dim: int) -> None:
        """Idempotently create the index schema (safe to re-run)."""
        schema = _build_index_schema(self.index_name, embedding_dim)
        self._index_client.create_or_update_index(schema)
        logger.info("Index '%s' created/updated (embedding_dim=%d)", self.index_name, embedding_dim)

    def delete_index(self) -> None:
        """Permanently delete the index (use with caution)."""
        self._index_client.delete_index(self.index_name)
        logger.warning("Index '%s' deleted", self.index_name)

    # ------------------------------------------------------------------

    def upload_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> dict[str, int]:
        """
        Upload embedded chunks to the index in batches.
        Returns a summary dict with ``succeeded`` and ``failed`` counts.
        """
        if not embedded_chunks:
            logger.warning("No chunks to upload")
            return {"succeeded": 0, "failed": 0}

        docs = [_chunk_to_document(ec) for ec in embedded_chunks]

        succeeded = 0
        failed = 0

        for start in range(0, len(docs), self.batch_size):
            batch = docs[start : start + self.batch_size]
            try:
                results = self._search_client.upload_documents(documents=batch)
                for r in results:
                    if r.succeeded:
                        succeeded += 1
                    else:
                        failed += 1
                        logger.error("Failed to index doc %s: %s", r.key, r.error_message)
                logger.debug(
                    "Batch %d–%d: %d succeeded, %d failed",
                    start,
                    start + len(batch) - 1,
                    succeeded,
                    failed,
                )
            except Exception as exc:
                logger.error("Batch upload error at position %d: %s", start, exc, exc_info=True)
                failed += len(batch)

        logger.info(
            "Upload complete: %d succeeded, %d failed (total=%d)",
            succeeded, failed, len(docs),
        )
        return {"succeeded": succeeded, "failed": failed}

    # ------------------------------------------------------------------

    def get_document_count(self) -> int:
        """Return the current number of documents in the index."""
        result = self._search_client.get_document_count()
        return result

    def delete_documents_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a specific document (for re-ingestion)."""
        results = self._search_client.search(
            search_text="*",
            filter=f"doc_id eq '{doc_id}'",
            select=["id"],
            top=10000,
        )
        keys = [{"id": r["id"]} for r in results]
        if keys:
            self._search_client.delete_documents(documents=keys)
            logger.info("Deleted %d chunks for doc_id='%s'", len(keys), doc_id)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def index_chunks(
    embedded_chunks: list[EmbeddedChunk],
    search_endpoint: str,
    admin_key: str,
    index_name: str = INDEX_NAME_DEFAULT,
    embedding_dim: Optional[int] = None,
) -> dict[str, int]:
    """
    One-shot helper: create/update the index schema, then upload all chunks.
    """
    dim = embedding_dim or (embedded_chunks[0].embedding_dim if embedded_chunks else 1536)
    indexer = KnowledgeBaseIndexer(
        search_endpoint=search_endpoint,
        admin_key=admin_key,
        index_name=index_name,
    )
    indexer.create_or_update_index(embedding_dim=dim)
    return indexer.upload_chunks(embedded_chunks)