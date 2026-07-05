"""Retrieval suite: recall@k and MRR against gold sources.

Fixture: data/evals/rag_qa.jsonl —
{"question": ..., "gold_sources": ["projects/reparo.md", ...]}
A retrieval counts as a hit if the chunk's source file matches any gold source.
"""

from __future__ import annotations

from evals.common import load_jsonl
from src.config import load_config


def run(k: int | None = None) -> dict:
    from src.rag.retrieve import retrieve

    cfg = load_config("rag")
    k = k or cfg["retriever"]["top_k"]
    rows = load_jsonl("rag_qa.jsonl")
    if not rows:
        return {"skipped": "no rag_qa.jsonl"}

    recalls, rrs, misses = [], [], []
    for row in rows:
        chunks = retrieve(row["question"], k=k)
        gold = set(row["gold_sources"])
        got = [c.source for c in chunks]
        hit_rank = next((i + 1 for i, s in enumerate(got) if s in gold), None)
        recalls.append(1.0 if hit_rank else 0.0)
        rrs.append(1.0 / hit_rank if hit_rank else 0.0)
        if not hit_rank:
            misses.append({"question": row["question"], "gold": sorted(gold), "got": got})

    result = {
        "n": len(rows),
        "k": k,
        f"recall@{k}": round(sum(recalls) / len(recalls), 4),
        "mrr": round(sum(rrs) / len(rrs), 4),
    }
    if misses:
        result["misses"] = misses
    return result


def main() -> None:
    import json

    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
