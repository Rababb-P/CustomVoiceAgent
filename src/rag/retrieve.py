"""Retrieval interface used by the agent tools and evals.

Returns plain Chunk dataclasses (not LangChain Documents) so callers don't
depend on LangChain types. Optional cross-encoder rerank behind a config flag.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from src.config import load_config


@dataclass
class Chunk:
    text: str
    source: str
    heading_path: str
    score: float

    def cite(self) -> str:
        return f"{self.source}#{self.heading_path}" if self.heading_path else self.source


@functools.lru_cache(maxsize=1)
def _store():
    from src.rag.store import VectorStore

    return VectorStore()


@functools.lru_cache(maxsize=1)
def _reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def retrieve(query: str, k: int | None = None, config: dict | None = None) -> list[Chunk]:
    cfg = config or load_config("rag")
    r = cfg["retriever"]
    k = k or r["top_k"]

    # Over-fetch when reranking so the cross-encoder has candidates to reorder.
    fetch_k = k * 3 if r["rerank"]["enabled"] else k
    results = _store().search(query, k=fetch_k)

    chunks = [
        Chunk(
            text=doc.page_content,
            source=doc.metadata.get("source", ""),
            heading_path=doc.metadata.get("heading_path", ""),
            score=score,
        )
        for doc, score in results
        if score >= r["score_threshold"]
    ]

    if r["rerank"]["enabled"] and chunks:
        ce = _reranker(r["rerank"]["model"])
        scores = ce.predict([(query, c.text) for c in chunks])
        for c, s in zip(chunks, scores, strict=True):
            c.score = float(s)
        chunks.sort(key=lambda c: c.score, reverse=True)
        chunks = chunks[: r["rerank"]["top_n"]]

    return chunks[:k]


def list_sources() -> list[str]:
    """Corpus table of contents: unique source#heading pairs, for the list_topics tool."""
    seen: dict[str, None] = {}
    for meta in _store().all_metadata():
        key = f"{meta.get('source', '')} — {meta.get('heading_path', '')}".strip(" —")
        seen.setdefault(key, None)
    return sorted(seen)
