"""Build the vector index from data/corpus markdown files.

Each file is split on markdown headers (heading path preserved in metadata),
then recursively into ~300-token chunks with 50-token overlap. Frontmatter keys
(source, topic, last_updated) are carried into every chunk's metadata.

Idempotent: the collection is reset and rebuilt on every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ROOT, load_config

# ~4 chars per token is close enough for chunk sizing on English prose.
CHARS_PER_TOKEN = 4


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal ---key: value--- frontmatter parser; returns (meta, body)."""
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4 :].lstrip("\n")
    return meta, text


def load_and_split(corpus_dir: Path, cfg: dict) -> list:
    from langchain_core.documents import Document
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    sp = cfg["splitter"]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[tuple(h) for h in sp["headers"]], strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=sp["chunk_tokens"] * CHARS_PER_TOKEN,
        chunk_overlap=sp["chunk_overlap_tokens"] * CHARS_PER_TOKEN,
    )

    docs: list[Document] = []
    for path in sorted(corpus_dir.rglob("*.md")):
        if path.name == "README.md":
            continue
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(corpus_dir)).replace("\\", "/")
        for section in header_splitter.split_text(body):
            heading_path = " > ".join(
                v for k, v in section.metadata.items() if k in ("h1", "h2", "h3")
            )
            for chunk in char_splitter.split_documents([section]):
                chunk.metadata = {
                    "source": rel,
                    "heading_path": heading_path,
                    "topic": meta.get("topic", ""),
                    "last_updated": meta.get("last_updated", ""),
                }
                docs.append(chunk)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    corpus_dir = ROOT / cfg["corpus_dir"]
    docs = load_and_split(corpus_dir, cfg)
    if not docs:
        raise SystemExit(f"No markdown found in {corpus_dir}. Write the corpus first.")

    from src.rag.store import VectorStore

    store = VectorStore(cfg)
    store.reset()
    n = store.add_documents(docs)
    sources = sorted({d.metadata["source"] for d in docs})
    print(f"Indexed {n} chunks from {len(sources)} files:")
    for s in sources:
        print(f"  {s}")


if __name__ == "__main__":
    main()
