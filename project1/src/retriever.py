"""
Enterprise-RAG: Hybrid retriever with ANN + BM25 fusion and DashScope Reranker.
"""
import time
from typing import Any

import numpy as np
from loguru import logger

from src.config import config
from src.embedder import Embedder


class DashScopeReranker:
    """DashScope reranker via HTTP API."""

    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name

    def compute_scores(self, query: str, documents: list[str], top_n: int = 5) -> list[float]:
        """Compute relevance scores for documents against query."""
        import httpx

        payload = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {"top_n": top_n},
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = httpx.post(
                    self.api_base,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("output", {}).get("results", [])
                # Build score array aligned with input documents order
                score_map = {r["index"]: r["relevance_score"] for r in results}
                return [score_map.get(i, 0.0) for i in range(len(documents))]
            except Exception as e:
                logger.warning(f"Reranker API attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Reranker API failed after {max_retries} attempts: {e}")

        return [0.0] * len(documents)


class Retriever:
    """Hybrid retrieval: Dense ANN + BM25 + Reranker."""

    def __init__(self, embedder: Embedder, vector_store: Any = None):
        cfg = config.get("retrieval", {})
        self.embedder = embedder
        self.vector_store = vector_store

        # Fusion weights
        self.vector_weight = cfg.get("vector_weight", 0.7)
        self.bm25_weight = cfg.get("bm25_weight", 0.3)
        self.candidate_top_k = cfg.get("candidate_top_k", 20)
        self.final_top_k = cfg.get("final_top_k", 5)

        # Reranker
        reranker_cfg = config.get("reranker", {})
        self.reranker_provider = reranker_cfg.get("provider", "dashscope")
        self.reranker_model_name = reranker_cfg.get("model_name", "gte-rerank")
        self.reranker_api_base = reranker_cfg.get(
            "api_base",
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        )
        self.reranker_api_key = reranker_cfg.get("api_key", "")
        self.reranker_top_n = reranker_cfg.get("top_n", 5)
        self._reranker: DashScopeReranker | None = None

        # Document store for text/metadata lookup
        self._dense_vectors: np.ndarray | None = None
        self._chunks_metadata: list[dict] = []
        self._chunks_text: list[str] = []

    @property
    def reranker(self) -> DashScopeReranker:
        """Lazy init the DashScope reranker."""
        if self._reranker is None:
            logger.info(f"Initializing DashScope reranker: {self.reranker_model_name}")
            self._reranker = DashScopeReranker(
                api_base=self.reranker_api_base,
                api_key=self.reranker_api_key,
                model_name=self.reranker_model_name,
            )
        return self._reranker

    def index(self, dense_vectors: np.ndarray, chunks_metadata: list[dict], chunks_text: list[str]) -> None:
        """Index the chunk vectors and metadata for retrieval."""
        self._dense_vectors = dense_vectors
        self._chunks_metadata = chunks_metadata
        self._chunks_text = chunks_text

        if self.vector_store is not None:
            self._index_to_vector_store(dense_vectors, chunks_metadata, chunks_text)

    def _index_to_vector_store(
        self,
        vectors: np.ndarray,
        metadata: list[dict],
        texts: list[str],
    ) -> None:
        """Insert vectors into Milvus/Chroma/Qdrant."""
        db_cfg = config.get("vector_db", {})
        backend = db_cfg.get("backend", "milvus")

        if backend == "milvus":
            self._index_milvus(vectors, metadata, texts)
        elif backend == "chroma":
            self._index_chroma(vectors, metadata, texts)
        elif backend == "qdrant":
            self._index_qdrant(vectors, metadata, texts)

    def _index_milvus(self, vectors, metadata, texts) -> None:
        """Insert into Milvus."""
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

            mc = config.get("vector_db", {}).get("milvus", {})
            host = mc.get("host", "localhost")
            port = mc.get("port", 19530)
            collection_name = mc.get("collection_name", "enterprise_knowledge")

            connections.connect(host=host, port=port)

            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1]),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="page", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, description="Enterprise RAG knowledge base")
            collection = Collection(collection_name, schema)

            # Insert data
            entities = []
            for i, (vec, meta, text) in enumerate(zip(vectors, metadata, texts)):
                entities.append({
                    "text": text[:65535],
                    "vector": vec.tolist(),
                    "source": meta.get("source", "")[:1024],
                    "page": meta.get("page", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                })
                if len(entities) >= 1000:
                    collection.insert(entities)
                    entities = []

            if entities:
                collection.insert(entities)

            # Create index
            index_params = {
                "metric_type": mc.get("metric_type", "IP"),
                "index_type": mc.get("index_type", "IVF_FLAT"),
                "params": {"nlist": 128},
            }
            collection.create_index("vector", index_params)
            collection.load()
            logger.info(f"Milvus: indexed {len(vectors)} vectors in '{collection_name}'")

        except ImportError:
            logger.warning("pymilvus not installed, skipping vector store indexing")
        except Exception as e:
            logger.warning(f"Milvus indexing failed (will use in-memory): {e}")

    def _index_chroma(self, vectors, metadata, texts) -> None:
        """Insert into Chroma."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            cc = config.get("vector_db", {}).get("chroma", {})
            persist_dir = cc.get("persist_directory", "./data/chroma_db")
            collection_name = cc.get("collection_name", "enterprise_knowledge")

            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            try:
                client.delete_collection(collection_name)
            except Exception:
                pass

            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            batch_size = 1000
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                collection.add(
                    embeddings=vectors[i:batch_end].tolist(),
                    documents=texts[i:batch_end],
                    metadatas=metadata[i:batch_end],
                    ids=[f"chunk_{j}" for j in range(i, batch_end)],
                )

            logger.info(f"Chroma: indexed {len(vectors)} vectors in '{collection_name}'")
        except ImportError:
            logger.warning("chromadb not installed, skipping vector store indexing")
        except Exception as e:
            logger.warning(f"Chroma indexing failed (will use in-memory): {e}")

    def _index_qdrant(self, vectors, metadata, texts) -> None:
        """Insert into Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams

            qc = config.get("vector_db", {}).get("qdrant", {})
            url = qc.get("url", "http://localhost:6333")
            collection_name = qc.get("collection_name", "enterprise_knowledge")
            vector_size = qc.get("vector_size", 1024)

            client = QdrantClient(url=url)

            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

            points = [
                PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload={
                        "text": text,
                        **meta,
                    },
                )
                for i, (vec, meta, text) in enumerate(zip(vectors, metadata, texts))
            ]

            batch_size = 500
            for i in range(0, len(points), batch_size):
                client.upsert(
                    collection_name=collection_name,
                    points=points[i:i + batch_size],
                )

            logger.info(f"Qdrant: indexed {len(vectors)} vectors in '{collection_name}'")
        except ImportError:
            logger.warning("qdrant_client not installed, skipping vector store indexing")
        except Exception as e:
            logger.warning(f"Qdrant indexing failed (will use in-memory): {e}")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Hybrid retrieval: ANN + BM25 fusion, then rerank.
        Returns list of dicts with keys: text, metadata, score, source_type.
        """
        top_k = top_k or self.final_top_k

        # Step 1: Dense retrieval (ANN)
        dense_results = self._dense_search(query, self.candidate_top_k)

        # Step 2: BM25 sparse retrieval
        bm25_results = self._bm25_search(query, self.candidate_top_k)

        # Step 3: Weighted fusion (RRF or linear combination)
        fused = self._reciprocal_rank_fusion(
            dense_results, bm25_results,
            k=60,
            dense_weight=self.vector_weight,
            sparse_weight=self.bm25_weight,
        )

        if not fused:
            return []

        # Step 4: Reranker
        reranked = self._rerank(query, fused, top_k=top_k)

        return reranked

    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """ANN search via cosine similarity on in-memory vectors."""
        if self._dense_vectors is None or len(self._dense_vectors) == 0:
            return []

        query_vec = self.embedder.embed_query(query)

        # Cosine similarity (vectors are already normalized)
        similarities = np.dot(self._dense_vectors, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """BM25 keyword search."""
        results = self.embedder.search_bm25(query, top_k=top_k)
        # Normalize BM25 scores to [0, 1]
        if not results:
            return []
        max_score = max(s for _, s in results) if results else 1.0
        return [(idx, score / max_score) for idx, score in results]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
        k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[tuple[int, float]]:
        """Weighted fusion of dense and sparse search results."""
        scores: dict[int, float] = {}

        # Dense contribution
        for rank, (idx, score) in enumerate(dense_results, start=1):
            scores[idx] = scores.get(idx, 0) + dense_weight * score

        # Sparse contribution
        for rank, (idx, score) in enumerate(sparse_results, start=1):
            scores[idx] = scores.get(idx, 0) + sparse_weight * score

        # Sort by fused score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:self.candidate_top_k]

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using DashScope reranker API."""
        valid_candidates = []
        documents = []

        for idx, _ in candidates:
            text = self.embedder.get_chunk_text(idx)
            if text:
                documents.append(text)
                valid_candidates.append(idx)

        if not documents:
            return []

        # Get reranker scores from DashScope API
        try:
            rerank_scores = self.reranker.compute_scores(
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
            )
        except Exception as e:
            logger.warning(f"Reranker failed, using fusion scores: {e}")
            return [
                {
                    "text": self.embedder.get_chunk_text(idx),
                    "metadata": self.embedder.get_chunk_metadata(idx),
                    "score": score,
                    "source_type": "fusion",
                }
                for idx, score in candidates[:top_k]
            ]

        # Sort by reranker score
        scored = list(zip(valid_candidates, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, rerank_score in scored[:top_k]:
            results.append({
                "text": self.embedder.get_chunk_text(idx),
                "metadata": self.embedder.get_chunk_metadata(idx),
                "score": float(rerank_score),
                "source_type": "reranked",
            })

        return results
