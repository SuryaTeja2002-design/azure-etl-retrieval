from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from openai import AzureOpenAI, RateLimitError, APIStatusError

from chunk import TextChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    """A TextChunk augmented with its dense vector embedding."""

    chunk: TextChunk
    embedding: list[float]          # Raw embedding vector
    model_name: str                 # e.g. "text-embedding-3-small"
    embedding_dim: int              # Dimension of the vector

    # Convenience accessors
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict:
        return self.chunk.metadata

    def as_numpy(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _retry_with_backoff(func, max_retries: int = 5, base_delay: float = 1.0):
    """
    Call ``func()`` and retry on rate-limit / transient errors using
    exponential back-off with jitter.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            wait = base_delay * (2 ** attempt) + (0.1 * attempt)
            logger.warning("Rate limited (attempt %d/%d) – sleeping %.1fs: %s", attempt + 1, max_retries, wait, e)
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code in (500, 502, 503, 504):
                wait = base_delay * (2 ** attempt)
                logger.warning("Transient API error %d – retrying in %.1fs", e.status_code, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Embedding call failed after {max_retries} retries")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _l2_normalise(vectors: list[list[float]]) -> list[list[float]]:
    """L2-normalise a batch of vectors. Zero-vectors are returned as-is."""
    result = []
    for v in vectors:
        arr = np.array(v, dtype=np.float64)
        norm = np.linalg.norm(arr)
        result.append((arr / norm).tolist() if norm > 0 else v)
    return result


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class ChunkEmbedder:
    """
    Embeds TextChunk objects using Azure OpenAI embedding models.

    Parameters
    ----------
    azure_endpoint : str
        Your Azure OpenAI resource endpoint, e.g.
        "https://my-resource.openai.azure.com/"
    api_key : str
        Azure OpenAI API key.
    deployment_name : str
        The deployment name (not the model name) configured in Azure OpenAI Studio.
        Defaults to "text-embedding-ada-002".
    api_version : str
        Azure OpenAI API version string.
    batch_size : int
        Number of chunks per API call.  Keep ≤ 2048 for ada-002.
    normalise : bool
        Whether to L2-normalise output vectors (recommended; default True).
    """

    DEFAULT_DEPLOYMENT = "text-embedding-ada-002"
    DEFAULT_API_VERSION = "2024-02-01"

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment_name: str = DEFAULT_DEPLOYMENT,
        api_version: str = DEFAULT_API_VERSION,
        batch_size: int = 100,
        normalise: bool = True,
    ) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.deployment_name = deployment_name
        self.batch_size = batch_size
        self.normalise = normalise
        self._embedding_dim: Optional[int] = None   # Resolved on first call

    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError("embedding_dim not yet resolved – call embed_chunks() first")
        return self._embedding_dim

    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string (useful for query-time embedding)."""
        result = _retry_with_backoff(
            lambda: self._client.embeddings.create(
                model=self.deployment_name,
                input=[text],
            )
        )
        vector = result.data[0].embedding
        if self._embedding_dim is None:
            self._embedding_dim = len(vector)
        if self.normalise:
            vector = _l2_normalise([vector])[0]
        return vector

    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        """
        Embed all chunks in batches, returning EmbeddedChunk objects.
        Logs progress every batch.
        """
        if not chunks:
            return []

        # Try tqdm for progress; fall back gracefully
        try:
            from tqdm import tqdm
            chunk_iter = tqdm(range(0, len(chunks), self.batch_size), desc="Embedding batches")
        except ImportError:
            chunk_iter = range(0, len(chunks), self.batch_size)

        embedded: list[EmbeddedChunk] = []

        for batch_start in chunk_iter:
            batch = chunks[batch_start : batch_start + self.batch_size]
            texts = [c.text for c in batch]

            result = _retry_with_backoff(
                lambda: self._client.embeddings.create(  # noqa: B023
                    model=self.deployment_name,
                    input=texts,
                )
            )

            vectors = [item.embedding for item in result.data]

            if self._embedding_dim is None:
                self._embedding_dim = len(vectors[0])

            if self.normalise:
                vectors = _l2_normalise(vectors)

            for chunk, vector in zip(batch, vectors):
                embedded.append(
                    EmbeddedChunk(
                        chunk=chunk,
                        embedding=vector,
                        model_name=self.deployment_name,
                        embedding_dim=len(vector),
                    )
                )

            logger.debug(
                "Embedded batch %d–%d / %d",
                batch_start,
                batch_start + len(batch) - 1,
                len(chunks),
            )

        logger.info(
            "Embedded %d chunks using '%s' (dim=%d)",
            len(embedded),
            self.deployment_name,
            self._embedding_dim,
        )
        return embedded


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[TextChunk],
    azure_endpoint: str,
    api_key: str,
    deployment_name: str = ChunkEmbedder.DEFAULT_DEPLOYMENT,
    batch_size: int = 100,
    normalise: bool = True,
) -> list[EmbeddedChunk]:
    """One-call helper that creates a ChunkEmbedder and embeds all chunks."""
    embedder = ChunkEmbedder(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        batch_size=batch_size,
        normalise=normalise,
    )
    return embedder.embed_chunks(chunks)