"""Generate the sentence pool for synthetic ASR training data.

Two sources into data/audio/synth/sentences.jsonl ({id, text, source, terms}):
1. LLM-written sentences that each contain at least one custom-vocab term, in
   varied speaking styles. Calls go through the cached src.llm wrapper (batched,
   flash-lite), so regeneration is free after the first run.
2. The Tech-Sentences-For-ASR-Training text dataset (general developer vocab).

Validation is strict: a claimed term must literally appear, sentences are
deduped case-insensitively, and length is bounded so TTS clips stay short.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

from src.config import ROOT, load_config

STYLES = [
    "a casual answer in a mock job interview",
    "a formal technical explanation",
    "a quick aside while pair programming",
    "a question an interviewer might ask",
    "an excited demo walkthrough",
]

_PROMPT = """Write {n} distinct English sentences for speech-recognition training.
Rules:
- Each sentence MUST naturally contain at least one of these terms, spelled exactly:
  {terms}
- Style: {style}. Spoken language, 8 to 25 words, no markdown, no quotes, no numbering.
- Vary sentence openings and which terms appear.

Output: one sentence per line, nothing else."""

MIN_WORDS, MAX_WORDS = 5, 30


def find_terms(text: str, vocab: list[str]) -> list[str]:
    return [t for t in vocab if re.search(re.escape(t), text, re.IGNORECASE)]


def validate(text: str, vocab: list[str], *, require_term: bool) -> list[str] | None:
    """Return matched terms if the sentence is usable, else None."""
    text = text.strip()
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return None
    if re.search(r"[#*`_\[\]{}|<>]", text):  # markdown / markup residue
        return None
    terms = find_terms(text, vocab)
    if require_term and not terms:
        return None
    return terms


def _sentence_id(text: str) -> str:
    return hashlib.sha256(text.lower().encode()).hexdigest()[:10]


def generate_llm_sentences(cfg: dict, *, generate_fn=None) -> list[dict]:
    """LLM sentences containing custom vocab. generate_fn is a test seam."""
    if generate_fn is None:
        from src.llm import generate as generate_fn  # noqa: PLC0415

    syn = cfg["data"]["synthetic"]
    vocab = cfg["data"]["custom_vocab"]
    target = syn["n_llm_sentences"]
    per_call = syn.get("sentences_per_call", 20)

    rows: dict[str, dict] = {}
    attempts = 0
    while len(rows) < target and attempts < (target // per_call) * 3 + 5:
        style = STYLES[attempts % len(STYLES)]
        # Salt the prompt per attempt so retries aren't cache hits of the same batch.
        prompt = _PROMPT.format(n=per_call, terms=", ".join(vocab), style=style)
        prompt += f"\n(batch {attempts})"
        attempts += 1
        for line in generate_fn(prompt, role="guard").splitlines():
            terms = validate(line, vocab, require_term=True)
            if terms is None:
                continue
            text = line.strip()
            rows.setdefault(
                _sentence_id(text),
                {"id": _sentence_id(text), "text": text, "source": "llm", "terms": terms},
            )
            if len(rows) >= target:
                break
    return list(rows.values())


def load_tech_sentences(cfg: dict) -> list[dict]:
    syn = cfg["data"]["synthetic"]["tech_sentences"]
    if not syn.get("enabled"):
        return []
    from datasets import load_dataset

    ds = load_dataset(syn["dataset"], split="train")
    vocab = cfg["data"]["custom_vocab"]
    rows: dict[str, dict] = {}
    text_col = next(c for c in ("text", "sentence", "sentences") if c in ds.column_names)
    for ex in ds:
        # Rows are multi-sentence paragraphs; split so clips stay short enough
        # for TTS and the word-count bounds in validate().
        for text in re.split(r"(?<=[.!?])\s+", str(ex[text_col]).strip()):
            _collect_tech_sentence(text, vocab, rows)
            if len(rows) >= syn.get("cap", 300):
                break
        if len(rows) >= syn.get("cap", 300):
            break
    return list(rows.values())


def _collect_tech_sentence(text: str, vocab: list[str], rows: dict[str, dict]) -> None:
    text = text.strip()
    if validate(text, vocab, require_term=False) is None:
        return
    rows.setdefault(
        _sentence_id(text),
        {
            "id": _sentence_id(text),
            "text": text,
            "source": "tech_sentences",
            "terms": find_terms(text, vocab),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr_finetune.yaml")
    parser.add_argument("--limit", type=int, help="override n_llm_sentences (smoke tests)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.limit:
        cfg["data"]["synthetic"]["n_llm_sentences"] = args.limit
        cfg["data"]["synthetic"]["tech_sentences"]["cap"] = args.limit

    llm_rows = generate_llm_sentences(cfg)
    tech_rows = load_tech_sentences(cfg)
    all_rows = llm_rows + [r for r in tech_rows if r["id"] not in {x["id"] for x in llm_rows}]

    out_dir = ROOT / cfg["data"]["synthetic"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sentences.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    covered = {t for r in all_rows for t in r["terms"]}
    missing = set(cfg["data"]["custom_vocab"]) - covered
    print(f"Wrote {len(all_rows)} sentences ({len(llm_rows)} llm, {len(tech_rows)} tech) -> {out}")
    if missing:
        print(f"WARNING: no sentences contain: {sorted(missing)} — raise n_llm_sentences.")


if __name__ == "__main__":
    main()
