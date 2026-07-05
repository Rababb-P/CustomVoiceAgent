"""ASR suite: WER/CER base vs fine-tuned on the held-out personal set, plus
per-term accuracy on custom vocab (the words base Whisper mangles).

Fixture: data/evals/asr_eval.jsonl — {"audio_path": ..., "transcript": ...}
"""

from __future__ import annotations

import re
from pathlib import Path

from evals.common import load_jsonl
from src.config import load_config


def _term_accuracy(references: list[str], hypotheses: list[str], terms: list[str]) -> dict:
    """Of reference sentences containing each term, what fraction of hypotheses got it."""
    out = {}
    for term in terms:
        pat = re.compile(re.escape(term), re.IGNORECASE)
        hits = total = 0
        for ref, hyp in zip(references, hypotheses, strict=True):
            if pat.search(ref):
                total += 1
                hits += bool(pat.search(hyp))
        if total:
            out[term] = round(hits / total, 3)
    return out


def _transcribe_all(model_name_or_dir: str, files: list[str]) -> list[str]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name_or_dir, compute_type="auto")
    hyps = []
    for f in files:
        segments, _ = model.transcribe(f, language="en")
        hyps.append(" ".join(s.text.strip() for s in segments))
    return hyps


def run() -> dict:
    import jiwer

    rows = [r for r in load_jsonl("asr_eval.jsonl") if Path(r["audio_path"]).exists()]
    if not rows:
        return {"skipped": "no asr_eval.jsonl rows with existing audio files"}

    cfg = load_config("asr_finetune")
    agent_cfg = load_config("agent")["server"]
    files = [r["audio_path"] for r in rows]
    refs = [r["transcript"] for r in rows]

    base_name = cfg["model"]["base"].split("/")[-1].replace("whisper-", "")
    results: dict = {"n": len(rows)}
    for label, model_id in [("base", base_name), ("finetuned", agent_cfg["asr_model_dir"])]:
        if label == "finetuned" and not Path(model_id).exists():
            results["finetuned"] = {"skipped": f"{model_id} not found — run make export-asr"}
            continue
        hyps = _transcribe_all(model_id, files)
        results[label] = {
            "wer": round(jiwer.wer(refs, hyps), 4),
            "cer": round(jiwer.cer(refs, hyps), 4),
            "term_accuracy": _term_accuracy(refs, hyps, cfg["data"]["custom_vocab"]),
        }
    return results


def main() -> None:
    import json

    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
