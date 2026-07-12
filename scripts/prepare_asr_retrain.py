"""Rebuild the recovery dataset using local text, Kokoro, and Common Voice.

Run from the repo root: python -m scripts.prepare_asr_retrain
No LLM API credentials are required. Audio and datasets remain gitignored.
"""

import json

import torch

from src.asr.gen_sentences import _sentence_id, find_terms, validate
from src.asr.prepare_data import _audio_features, _load_common_voice, _read_audio, _save_dataset
from src.asr.synthesize import plan_renders, synthesize_all
from src.config import ROOT, load_config


def main():
    from datasets import Dataset, concatenate_datasets

    torch.set_num_threads(4)
    cfg = load_config("asr_retrain_cpu")
    syn = cfg["data"]["synthetic"]
    vocab = cfg["data"]["custom_vocab"]
    texts = (ROOT / "docs/asr_retrain_sentences.txt").read_text(encoding="utf-8").splitlines()
    rows = []
    for text in texts:
        if validate(text, vocab, require_term=False) is None:
            raise ValueError(f"Invalid sentence: {text}")
        rows.append(
            {
                "id": _sentence_id(text),
                "text": text,
                "source": "assistant_authored",
                "terms": find_terms(text, vocab),
            }
        )
    renders = plan_renders(rows, syn)
    for split in ("train", "val"):
        covered = {
            term
            for row in renders
            if row["split"] == split
            for term in find_terms(row["text"], vocab)
        }
        if covered != set(vocab):
            raise ValueError(f"Missing vocabulary in {split}: {set(vocab) - covered}")
    out_dir = ROOT / syn["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sentences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"Rendering {len(renders)} clips from {len(rows)} sentences.", flush=True)
    manifest = synthesize_all(renders, out_dir, syn)
    (out_dir / "synth_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
    )

    def dataset_for(split):
        def generate():
            for row in manifest:
                if row["split"] == split:
                    yield {
                        "audio": {
                            "array": _read_audio(str(ROOT / row["audio_path"]), 16000),
                            "sampling_rate": 16000,
                        },
                        "transcript": row["text"],
                    }

        return Dataset.from_generator(generate, features=_audio_features())

    train, val = dataset_for("train"), dataset_for("val")
    cv = _load_common_voice(cfg, len(train))
    train = concatenate_datasets([train, cv]).shuffle(seed=42)
    _save_dataset(train, val, cfg)
    print(f"Done: {len(train)} train, {len(val)} validation; all eight terms covered.", flush=True)


if __name__ == "__main__":
    main()
