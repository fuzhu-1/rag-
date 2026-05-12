"""
Enterprise-RAG: End-to-end RAG pipeline orchestrating all modules.
"""
import os
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import config
from src.embedder import Embedder
from src.generator import Generator
from src.loader import DocumentLoader
from src.retriever import Retriever
from src.splitter import TextSplitter


class RAGPipeline:
    """End-to-end RAG pipeline: load → split → embed → retrieve → generate."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter()
        self.embedder = Embedder()
        self.generator = Generator()
        self.retriever = Retriever(self.embedder)

        self._indexed = False
        self._chunks: list = []

    def ingest_directory(self, directory: str | None = None) -> int:
        """
        Ingest all documents from a directory:
        load → split → embed → index
        Returns number of chunks indexed.
        """
        # Load documents
        documents = self.loader.load_directory(directory)
        if not documents:
            logger.warning("No documents found to ingest")
            return 0

        # Split into chunks
        chunks = self.splitter.split(documents)
        if not chunks:
            logger.warning("No chunks produced after splitting")
            return 0

        # Embed and index
        dense_vectors, metadata = self.embedder.embed_documents(chunks)
        chunk_texts = [c.content for c in chunks]

        self.retriever.index(dense_vectors, metadata, chunk_texts)
        self._chunks = chunks
        self._indexed = True

        # Persist BM25 index
        bm25_path = Path(config["project"]["data_dir"]).parent / "bm25_index.pkl"
        self.embedder.save_bm25(str(bm25_path))

        logger.info(f"Ingestion complete: {len(documents)} docs → {len(chunks)} chunks")
        return len(chunks)

    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file."""
        documents = self.loader.load_file(file_path)
        if not documents:
            return 0

        chunks = self.splitter.split(documents)
        dense_vectors, metadata = self.embedder.embed_documents(chunks)
        chunk_texts = [c.content for c in chunks]

        self.retriever.index(dense_vectors, metadata, chunk_texts)
        self._chunks = chunks
        self._indexed = True

        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_cot: bool = True,
        history: list[dict[str, str]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Any:
        """
        Run the full RAG pipeline for a query.

        Args:
            question: User question
            top_k: Number of contexts to retrieve
            use_cot: Use Chain-of-Thought reasoning
            history: Conversation history
            stream: Stream the answer (returns iterator)

        Returns:
            Dict with keys: answer, reasoning (if CoT), contexts, question
            Or streaming iterator if stream=True
        """
        if not self._indexed:
            return {
                "answer": "系统尚未索引任何文档，请先上传文档。",
                "contexts": [],
                "question": question,
            }

        # Query expansion (if enabled)
        queries = [question]
        if config.get("query_expansion", {}).get("enabled", False):
            variations = self.generator.expand_query(question)
            queries.extend(variations)
            logger.info(f"Query expanded to {len(queries)} variations")

        # Retrieve contexts for all query variations
        all_contexts: dict[str, dict] = {}
        for q in queries:
            contexts = self.retriever.retrieve(q, top_k=top_k)
            for ctx in contexts:
                key = ctx["text"][:100]  # Deduplicate by content
                if key not in all_contexts:
                    all_contexts[key] = ctx

        merged_contexts = list(all_contexts.values())[:top_k or 5]
        logger.info(f"Retrieved {len(merged_contexts)} unique contexts")

        # Generate answer
        if stream:
            return self.generator.generate(
                query=question,
                contexts=merged_contexts,
                history=history,
                stream=True,
            )

        if use_cot:
            result = self.generator.generate_with_cot(
                query=question,
                contexts=merged_contexts,
                history=history,
            )
            return {
                "answer": result.get("answer", ""),
                "reasoning": result.get("reasoning", ""),
                "contexts": merged_contexts,
                "question": question,
            }
        else:
            answer = self.generator.generate(
                query=question,
                contexts=merged_contexts,
                history=history,
            )
            return {
                "answer": answer,
                "contexts": merged_contexts,
                "question": question,
            }

    def get_chunks(self) -> list[dict]:
        """Get all indexed chunks with metadata."""
        return [
            {
                "text": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                "metadata": c.metadata,
                "doc_id": c.doc_id,
            }
            for c in self._chunks
        ]

    def delete_document(self, source_name: str) -> int:
        """Delete chunks by source document name."""
        before = len(self._chunks)
        self._chunks = [
            c for c in self._chunks
            if c.metadata.get("source") != source_name
        ]
        after = len(self._chunks)
        deleted = before - after

        if deleted > 0:
            # Re-index
            texts = [c.content for c in self._chunks]
            if texts:
                dense_vectors, metadata = self.embedder.embed_documents(self._chunks)
                self.retriever.index(dense_vectors, metadata, texts)

        logger.info(f"Deleted {deleted} chunks from '{source_name}'")
        return deleted


# Global pipeline instance
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def rag_pipeline(query: str, top_k: int = 5) -> dict[str, Any]:
    """Convenience function matching the spec requirement."""
    pipeline = get_pipeline()
    return pipeline.query(question=query, top_k=top_k)
