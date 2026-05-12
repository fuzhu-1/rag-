"""
Enterprise-RAG: Text splitter with recursive character splitting
and semantic (Markdown header) chunking support.
"""
import re
from typing import Any

from loguru import logger

from src.config import config


class TextSplitter:
    """Recursive character-based text splitter with Chinese-aware separators."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        cfg = config.get("splitter", {})
        self.chunk_size = chunk_size or cfg.get("chunk_size", 512)
        self.chunk_overlap = chunk_overlap or cfg.get("chunk_overlap", 100)
        self.separators = separators or cfg.get("separators", [
            "\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", " ",
        ])
        self.semantic_enabled = cfg.get("enable_semantic_chunking", False)

    def split(self, documents: list) -> list:
        """Split a list of Document objects into chunks."""
        from src.loader import Document

        all_chunks: list[Document] = []
        for doc in documents:
            chunks = self._split_text(doc.content)
            for i, chunk_text in enumerate(chunks):
                chunk_meta = {
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "doc_id": doc.doc_id,
                }
                all_chunks.append(Document(content=chunk_text, metadata=chunk_meta))

        logger.info(f"Split {len(documents)} docs into {len(all_chunks)} chunks")
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """Recursively split text by separators."""
        # Try semantic chunking first if enabled
        if self.semantic_enabled:
            chunks = self._semantic_split(text)
            if len(chunks) > 1:
                return self._merge_chunks(chunks)

        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Split text recursively using the given separators."""
        final_chunks: list[str] = []
        separator = separators[-1] if separators else " "

        # Try each separator
        for sep in separators:
            if not sep:
                continue
            if sep in text:
                splits = self._split_by_separator(text, sep)
                for split in splits:
                    if self._text_length(split) <= self.chunk_size:
                        if split.strip():
                            final_chunks.append(split.strip())
                    else:
                        # Recurse with remaining separators
                        remaining = separators[separators.index(sep) + 1:]
                        final_chunks.extend(self._recursive_split(split, remaining))
                break
        else:
            # No separator found, force split by character
            final_chunks = self._force_split(text)

        return final_chunks

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Split text by separator, preserving the separator."""
        if not separator:
            return [text]

        splits = text.split(separator)
        # Preserve separator on chunks (except last)
        result = []
        for i, s in enumerate(splits):
            if i < len(splits) - 1:
                result.append(s + separator)
            elif s.strip():
                result.append(s)
        return result

    def _force_split(self, text: str) -> list[str]:
        """Force split by character when no separator works."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def _semantic_split(self, text: str) -> list[str]:
        """Split text by Markdown headers (##, ###, etc.)."""
        # Split by markdown headers
        header_pattern = r"(#{1,6}\s+.+?)(?=\n#{1,6}\s+|\Z)"
        sections = re.findall(header_pattern, text, re.DOTALL)

        if len(sections) <= 1:
            # Try splitting by double newlines as fallback
            sections = text.split("\n\n")

        return [s.strip() for s in sections if s.strip()]

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks that are too small, split chunks that are too large."""
        result: list[str] = []
        buffer = ""

        for chunk in chunks:
            if self._text_length(buffer) + self._text_length(chunk) <= self.chunk_size:
                buffer = (buffer + "\n" + chunk).strip() if buffer else chunk
            else:
                if buffer:
                    if self._text_length(buffer) <= self.chunk_size:
                        result.append(buffer)
                    else:
                        result.extend(self._recursive_split(buffer, self.separators))
                buffer = chunk

        if buffer:
            if self._text_length(buffer) <= self.chunk_size:
                result.append(buffer)
            else:
                result.extend(self._recursive_split(buffer, self.separators))

        return result

    @staticmethod
    def _text_length(text: str) -> int:
        """Count characters (Chinese-aware: each CJK char counts as 1)."""
        return len(text)
