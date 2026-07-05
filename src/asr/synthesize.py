"""Render the sentence pool into multi-voice training audio.

sentences.jsonl -> data/audio/synth/wavs/*.wav + synth_manifest.jsonl
({id, audio_path, text, voice, speed, noisy, split}).

Split discipline (the part that makes val WER meaningful):
- A heldout_sentence_fraction of sentences is reserved for validation.
- Validation renders use ONLY heldout_voices; training renders use ONLY
  train_voices. Val therefore measures generalization to unseen sentences
  spoken by unseen voices, not memorization of either.
- Augmentation (speed perturb + additive noise) applies to train renders only.

Voice assignment is deterministic (hash of sentence id) so re-runs are stable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import ROOT, load_config

TARGET_SR = 16000


def split_sentences(rows: list[dict], heldout_fraction: float) -> dict[str, str]:
    """sentence id -> 'train' | 'val'. Deterministic via id hash order."""
    ordered = sorted(rows, key=lambda r: r["id"])
    n_val = max(1, int(len(ordered) * heldout_fraction)) if ordered else 0
    val_ids = {r["id"] for r in ordered[:n_val]}
    return {r["id"]: ("val" if r["id"] in val_ids else "train") for r in ordered}


def assign_voices(sentence_id: str, voices: list[str], per_sentence: int = 2) -> list[str]:
    """Deterministically pick `per_sentence` voices for a sentence."""
    start = int(sentence_id, 16) % len(voices)
    return [voices[(start + i) % len(voices)] for i in range(min(per_sentence, len(voices)))]


def add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise at the given SNR."""
    signal_power = float(np.mean(wav**2)) or 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10))
    return wav + rng.normal(0, np.sqrt(noise_power), wav.shape).astype(wav.dtype)


def plan_renders(rows: list[dict], syn_cfg: dict) -> list[dict]:
    """Pure planning step (unit-testable): which (sentence, voice, speed, noise)
    combinations to render, with train/val separation enforced."""
    splits = split_sentences(rows, syn_cfg.get("heldout_sentence_fraction", 0.1))
    rng = np.random.default_rng(42)
    speeds = syn_cfg.get("speeds", [1.0])
    noise_fraction = syn_cfg.get("noise_fraction", 0.0)

    renders = []
    for row in rows:
        split = splits[row["id"]]
        if split == "train":
            voices = assign_voices(row["id"], syn_cfg["train_voices"])
            for voice in voices:
                speed = speeds[int(row["id"], 16) % len(speeds)]
                renders.append(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "voice": voice,
                        "speed": speed,
                        "noisy": bool(rng.random() < noise_fraction),
                        "split": "train",
                    }
                )
        else:
            for voice in assign_voices(row["id"], syn_cfg["heldout_voices"], per_sentence=1):
                renders.append(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "voice": voice,
                        "speed": 1.0,
                        "noisy": False,
                        "split": "val",
                    }
                )
    return renders


def synthesize_all(renders: list[dict], out_dir: Path, syn_cfg: dict) -> list[dict]:
    import librosa
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    snr_lo, snr_hi = syn_cfg.get("noise_snr_db", [15, 30])

    manifest = []
    for i, r in enumerate(renders):
        path = wav_dir / f"{r['id']}_{r['voice']}_{r['speed']:.1f}{'_n' if r['noisy'] else ''}.wav"
        if not path.exists():
            chunks = [np.asarray(audio) for _, _, audio in pipeline(r["text"], voice=r["voice"])]
            wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            wav = librosa.resample(wav.astype(np.float32), orig_sr=24000, target_sr=TARGET_SR)
            if r["speed"] != 1.0:
                wav = librosa.effects.time_stretch(wav, rate=r["speed"])
            if r["noisy"]:
                wav = add_noise(wav, rng.uniform(snr_lo, snr_hi), rng)
            sf.write(path, np.clip(wav, -1.0, 1.0), TARGET_SR)
        manifest.append({**r, "audio_path": str(path.relative_to(ROOT)).replace("\\", "/")})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(renders)} rendered")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr_finetune.yaml")
    parser.add_argument("--limit", type=int, help="render only the first N sentences (smoke)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    syn_cfg = cfg["data"]["synthetic"]

    out_dir = ROOT / syn_cfg["out_dir"]
    sentences_path = out_dir / "sentences.jsonl"
    if not sentences_path.exists():
        raise SystemExit(f"{sentences_path} missing — run python -m src.asr.gen_sentences first")
    rows = [json.loads(line) for line in sentences_path.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = sorted(rows, key=lambda r: r["id"])[: args.limit]

    renders = plan_renders(rows, syn_cfg)
    print(f"Planned {len(renders)} renders from {len(rows)} sentences "
          f"({sum(1 for r in renders if r['split'] == 'val')} val).")
    manifest = synthesize_all(renders, out_dir, syn_cfg)

    manifest_path = out_dir / "synth_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
