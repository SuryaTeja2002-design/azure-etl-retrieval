

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

from extract import DocumentExtractor, ExtractedDocument
from chunk import DocumentChunker, TextChunk
from embed import ChunkEmbedder, EmbeddedChunk
from index import KnowledgeBaseIndexer, INDEX_NAME_DEFAULT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    Full document ingestion pipeline: Blob Storage → Azure AI Search.

    Parameters
    ----------
    blob_connection_string : str
    container_name : str
    search_endpoint : str
    search_admin_key : str
    openai_endpoint : str
    openai_key : str
    embedding_deployment : str
    index_name : str
    doc_intelligence_endpoint : str | None
    doc_intelligence_key : str | None
    target_tokens : int
    overlap_tokens : int
    batch_size_embed : int
    batch_size_index : int
    """

    def __init__(
        self,
        blob_connection_string: str,
        container_name: str,
        search_endpoint: str,
        search_admin_key: str,
        openai_endpoint: str,
        openai_key: str,
        embedding_deployment: str = "text-embedding-ada-002",
        index_name: str = INDEX_NAME_DEFAULT,
        doc_intelligence_endpoint: Optional[str] = None,
        doc_intelligence_key: Optional[str] = None,
        target_tokens: int = 512,
        overlap_tokens: int = 50,
        batch_size_embed: int = 100,
        batch_size_index: int = 500,
    ) -> None:
        self.extractor = DocumentExtractor(
            blob_connection_string=blob_connection_string,
            container_name=container_name,
            doc_intelligence_endpoint=doc_intelligence_endpoint,
            doc_intelligence_key=doc_intelligence_key,
        )
        self.chunker = DocumentChunker(
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        self.embedder = ChunkEmbedder(
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
            deployment_name=embedding_deployment,
            batch_size=batch_size_embed,
            normalise=True,
        )
        self.indexer = KnowledgeBaseIndexer(
            search_endpoint=search_endpoint,
            admin_key=search_admin_key,
            index_name=index_name,
            batch_size=batch_size_index,
        )

    # ------------------------------------------------------------------

    def run(
        self,
        blob_prefix: str = "",
        blob_names: Optional[list[str]] = None,
        force_reindex: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """
        Execute the full pipeline.

        Parameters
        ----------
        blob_prefix : str
            Process only blobs whose name starts with this prefix.
        blob_names : list[str] | None
            If supplied, process only these specific blobs (overrides blob_prefix).
        force_reindex : bool
            If True, delete existing index entries for each document before re-ingesting.
        dry_run : bool
            Extract and chunk without embedding or indexing – useful for cost estimation.

        Returns
        -------
        dict with pipeline statistics.
        """
        t_start = time.monotonic()
        stats: dict = {
            "blobs_discovered": 0,
            "blobs_extracted": 0,
            "chunks_produced": 0,
            "chunks_embedded": 0,
            "chunks_indexed": 0,
            "index_failed": 0,
            "elapsed_seconds": 0.0,
        }

        # 1. Discover blobs -----------------------------------------------
        if blob_names:
            target_blobs = blob_names
        else:
            target_blobs = self.extractor.list_blobs(prefix=blob_prefix)

        stats["blobs_discovered"] = len(target_blobs)
        if not target_blobs:
            logger.warning("No supported blobs found – nothing to ingest")
            return stats

        # 2. Extract -------------------------------------------------------
        logger.info("=== STAGE 1: Extraction (%d blobs) ===", len(target_blobs))
        documents: list[ExtractedDocument] = []
        for name in target_blobs:
            try:
                doc = self.extractor.extract_blob(name)
                documents.append(doc)
                stats["blobs_extracted"] += 1
            except Exception as exc:
                logger.error("Extraction failed for '%s': %s", name, exc, exc_info=True)

        # 3. Chunk ---------------------------------------------------------
        logger.info("=== STAGE 2: Chunking (%d documents) ===", len(documents))
        all_chunks: list[TextChunk] = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            logger.info("  %s → %d chunks", doc.doc_id, len(chunks))

        stats["chunks_produced"] = len(all_chunks)

        if dry_run:
            logger.info("DRY RUN – skipping embedding and indexing")
            stats["elapsed_seconds"] = time.monotonic() - t_start
            return stats

        # 4. Embed ---------------------------------------------------------
        logger.info("=== STAGE 3: Embedding (%d chunks) ===", len(all_chunks))
        embedded: list[EmbeddedChunk] = self.embedder.embed_chunks(all_chunks)
        stats["chunks_embedded"] = len(embedded)

        # 5. Create / update index schema ----------------------------------
        logger.info("=== STAGE 4: Indexing ===")
        if embedded:
            self.indexer.create_or_update_index(embedding_dim=embedded[0].embedding_dim)

        # Optional: purge stale chunks for documents being re-indexed
        if force_reindex:
            processed_doc_ids = {doc.doc_id for doc in documents}
            for doc_id in processed_doc_ids:
                self.indexer.delete_documents_by_doc_id(doc_id)

        # 6. Upload --------------------------------------------------------
        result = self.indexer.upload_chunks(embedded)
        stats["chunks_indexed"] = result["succeeded"]
        stats["index_failed"] = result["failed"]

        stats["elapsed_seconds"] = round(time.monotonic() - t_start, 2)
        logger.info("=== Pipeline complete in %.1fs ===", stats["elapsed_seconds"])
        logger.info("Stats: %s", stats)
        return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RAG ingestion pipeline: Blob Storage → Azure AI Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required Azure credentials (fall back to env vars)
    p.add_argument("--blob-conn-str", default=os.getenv("AZURE_BLOB_CONN_STR"), required=not os.getenv("AZURE_BLOB_CONN_STR"))
    p.add_argument("--container", default=os.getenv("AZURE_BLOB_CONTAINER", "knowledge-base"))
    p.add_argument("--search-endpoint", default=os.getenv("AZURE_SEARCH_ENDPOINT"), required=not os.getenv("AZURE_SEARCH_ENDPOINT"))
    p.add_argument("--search-admin-key", default=os.getenv("AZURE_SEARCH_ADMIN_KEY"), required=not os.getenv("AZURE_SEARCH_ADMIN_KEY"))
    p.add_argument("--openai-endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"), required=not os.getenv("AZURE_OPENAI_ENDPOINT"))
    p.add_argument("--openai-key", default=os.getenv("AZURE_OPENAI_KEY"), required=not os.getenv("AZURE_OPENAI_KEY"))

    # Optional
    p.add_argument("--embedding-deployment", default=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"))
    p.add_argument("--index-name", default=INDEX_NAME_DEFAULT)
    p.add_argument("--blob-prefix", default="", help="Process only blobs under this prefix")
    p.add_argument("--doc-intelligence-endpoint", default=os.getenv("AZURE_DOC_INTEL_ENDPOINT"))
    p.add_argument("--doc-intelligence-key", default=os.getenv("AZURE_DOC_INTEL_KEY"))

    # Chunking parameters
    p.add_argument("--target-tokens", type=int, default=512)
    p.add_argument("--overlap-tokens", type=int, default=50)

    # Behaviour flags
    p.add_argument("--force-reindex", action="store_true", help="Delete existing index entries before re-ingesting")
    p.add_argument("--dry-run", action="store_true", help="Extract+chunk only; skip embedding/indexing")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main(argv=None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    pipeline = IngestionPipeline(
        blob_connection_string=args.blob_conn_str,
        container_name=args.container,
        search_endpoint=args.search_endpoint,
        search_admin_key=args.search_admin_key,
        openai_endpoint=args.openai_endpoint,
        openai_key=args.openai_key,
        embedding_deployment=args.embedding_deployment,
        index_name=args.index_name,
        doc_intelligence_endpoint=args.doc_intelligence_endpoint,
        doc_intelligence_key=args.doc_intelligence_key,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    stats = pipeline.run(
        blob_prefix=args.blob_prefix,
        force_reindex=args.force_reindex,
        dry_run=args.dry_run,
    )

    print("\n=== Ingestion Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()