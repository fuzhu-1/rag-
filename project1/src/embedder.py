"""
Enterprise-RAG: Embedding module using 阿里百炼 DashScope API (OpenAI 兼容).
Also builds BM25 sparse index for hybrid retrieval.
"""
import pickle
import time
from pathlib import Path

import numpy as np
from loguru import logger
from openai import OpenAI
from rank_bm25 import BM25Okapi

from src.config import config


class Embedder:
    """Embedding via DashScope compatible API + BM25 sparse index."""

    def __init__(self):
        cfg = config.get("embedding", {})
        self.provider = cfg.get("provider", "dashscope")
        self.model_name = cfg.get("model_name", "text-embedding-v1")
        self.api_base = cfg.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.api_key = cfg.get("api_key", "")
        self.normalize = cfg.get("normalize", True)
        self.dense_dim = cfg.get("dense_dim", 1536)
        self.batch_size = cfg.get("batch_size", 25)

        self._client: OpenAI | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[str] = []
        self._chunk_metadata: list[dict] = []

    @property
    def client(self) -> OpenAI:
        """Lazy init the OpenAI-compatible client pointing to DashScope."""
        if self._client is None:
            logger.info(f"Embedding API: {self.api_base}  model={self.model_name}")
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def embed_documents(self, chunks: list) -> tuple[np.ndarray, list[dict]]:
        """
        Embed a list of Document chunks via DashScope API.

        Returns:
            dense_vectors: numpy array of shape (n_chunks, dim)
            metadata: list of metadata dicts
        """
        texts = [chunk.content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        if not texts:
            logger.warning("No texts to embed")
            return np.empty((0, self.dense_dim)), []

        # Generate dense embeddings via API (with batching)
        logger.info(f"Embedding {len(texts)} chunks via DashScope ({self.model_name})...")
        all_vectors: list[np.ndarray] = []

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            logger.info(f"  batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")

            vectors = self._call_embedding_api(batch_texts)
            all_vectors.extend(vectors)

            if batch_num < total_batches:
                time.sleep(0.5)  # rate limiting

        dense_vectors = np.array(all_vectors, dtype=np.float32)

        # Normalize if configured
        if self.normalize:
            norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # avoid div by zero
            dense_vectors = dense_vectors / norms

        # Auto-detect dimension from first result
        if dense_vectors.shape[1] != self.dense_dim:
            self.dense_dim = dense_vectors.shape[1]
            logger.info(f"Detected embedding dim: {self.dense_dim}")

        # Build BM25 index
        logger.info("Building BM25 index...")
        tokenized = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_chunks = texts
        self._chunk_metadata = metadata

        logger.info(f"Embedded {len(texts)} chunks → dim={dense_vectors.shape[1]}")
        return dense_vectors, metadata

    def _call_embedding_api(self, texts: list[str]) -> list[np.ndarray]:
        """Call DashScope embedding API (OpenAI-compatible)."""
        # Ensure all inputs are non-empty strings
        clean_texts = [t if t.strip() else " " for t in texts]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.client.embeddings.create(
                    model=self.model_name,
                    input=clean_texts,
                )
                # OpenAI-compatible response: resp.data[i].embedding
                vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
                return vectors

            except Exception as e:
                logger.warning(f"Embedding API attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Embedding API failed after {max_retries} attempts: {e}")

        return []

    def embed_query(self, query: str) -> np.ndarray:
        """Generate dense embedding for a single query string."""
        vectors = self._call_embedding_api([query])
        vec = vectors[0]
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def search_bm25(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """BM25 keyword search. Returns list of (chunk_index, score)."""
        if self._bm25 is None:
            logger.warning("BM25 index not built yet")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in indices if scores[idx] > 0]

    def get_chunk_text(self, index: int) -> str:
        """Get chunk text by index."""
        if 0 <= index < len(self._bm25_chunks):
            return self._bm25_chunks[index]
        return ""

    def get_chunk_metadata(self, index: int) -> dict:
        """Get chunk metadata by index."""
        if 0 <= index < len(self._chunk_metadata):
            return self._chunk_metadata[index]
        return {}

    def save_bm25(self, path: str) -> None:
        """Persist BM25 index to disk."""
        data = {"chunks": self._bm25_chunks, "metadata": self._chunk_metadata}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {path}")

    def load_bm25(self, path: str) -> None:
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25_chunks = data["chunks"]
        self._chunk_metadata = data["metadata"]
        tokenized = [self._tokenize(t) for t in self._bm25_chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index loaded from {path} ({len(self._bm25_chunks)} chunks)")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Character-level for CJK + word-level for English."""
        import re
        tokens = []
        for part in re.split(r"(\s+)", text):
            if part.strip():
                cjk = sum(1 for c in part if "一" <= c <= "鿿")
                if cjk > len(part) * 0.3:
                    tokens.extend(list(part))
                else:
                    tokens.extend(part.split())
        return [t.lower() for t in tokens if t.strip()]
