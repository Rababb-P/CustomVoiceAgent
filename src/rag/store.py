"""Thin wrapper around the Chroma vectorstore so the vector DB stays swappable.

Everything else in the project talks to VectorStore, never to Chroma directly.
"""

from __future__ import annotations

import functools

from src.config import ROOT, load_config


@functools.lru_cache(maxsize=1)
def _embeddings(model: str, device: str, normalize: bool):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )


class VectorStore:
    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config("rag")
        emb_cfg = self.cfg["embeddings"]
        self._embeddings = _embeddings(
            emb_cfg["model"], emb_cfg.get("device", "cpu"), emb_cfg.get("normalize", True)
        )
        from langchain_chroma import Chroma

        self._db = Chroma(
            collection_name=self.cfg["collection"],
            embedding_function=self._embeddings,
            persist_directory=str(ROOT / self.cfg["persist_dir"]),
        )

    def reset(self) -> None:
        """Drop and recreate the collection (makes ingest idempotent)."""
        self._db.reset_collection()

    def add_documents(self, docs: list) -> int:
        if docs:
            self._db.add_documents(docs)
        return len(docs)

    def search(self, query: str, k: int) -> list[tuple]:
        """Returns [(Document, similarity_score)] with score in [0, 1], best first."""
        results = self._db.similarity_search_with_relevance_scores(query, k=k)
        return sorted(results, key=lambda pair: pair[1], reverse=True)

    def count(self) -> int:
        return self._db._collection.count()

    def all_metadata(self) -> list[dict]:
        return self._db._collection.get(include=["metadatas"])["metadatas"] or []
