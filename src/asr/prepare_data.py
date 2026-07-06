"""Build the Whisper fine-tuning dataset.

Primary path (no recordings needed): --synthetic
    Assembles the HF DatasetDict from the synthetic multi-voice renders
    (src.asr.gen_sentences + src.asr.synthesize), a TechVoice real-speaker
    mix-in, and a Common Voice slice. Also regenerates data/evals/asr_eval.jsonl
    with real-speaker (TechVoice held-out) and cross-voice (synth val) rows.

Legacy path (your own voice):
    default:     segment raw recordings + bootstrap transcripts into a TSV
    --from-tsv:  build the dataset from the hand-corrected TSV
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config import ROOT, load_config

# --------------------------------------------------------------- shared helpers

# Audio is decoded with soundfile (wav + mp3) instead of datasets' Audio
# feature: datasets >=4 delegates decoding to torchcodec, which needs FFmpeg
# shared libraries — a system dependency this repo avoids. Columns are plain
# {array, sampling_rate} structs, which is exactly what train.py consumes.


def _audio_features():
    from datasets import Features, Sequence, Value

    return Features(
        {
            "audio": {"array": Sequence(Value("float32")), "sampling_rate": Value("int64")},
            "transcript": Value("string"),
        }
    )


def _read_audio(source, target_sr: int):
    """Decode a wav/mp3 file path or bytes to mono float32 at target_sr."""
    import io

    import librosa
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(source) if isinstance(source, bytes) else source,
                      dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.asarray(wav, dtype="float32")


def _load_common_voice(cfg: dict, n: int):
    """Streamed slice of Common Voice as a Dataset, or None if disabled."""
    cv_cfg = cfg["data"].get("common_voice", {})
    if not cv_cfg.get("enabled") or n <= 0:
        return None
    from datasets import Audio, Dataset, load_dataset

    sr = cfg["data"]["sample_rate"]
    cv = load_dataset(cv_cfg["dataset"], cv_cfg["language"], split="train", streaming=True)
    cv = cv.cast_column("audio", Audio(decode=False))  # raw bytes; we decode ourselves

    def gen():
        for ex in cv.take(n):
            yield {
                "audio": {"array": _read_audio(ex["audio"]["bytes"], sr), "sampling_rate": sr},
                "transcript": ex["sentence"],
            }

    ds = Dataset.from_generator(gen, features=_audio_features())
    print(f"Mixed in {len(ds)} Common Voice examples.")
    return ds


def _save_dataset(train, val, cfg: dict) -> None:
    from datasets import DatasetDict

    out = Path(cfg["data"]["processed_dir"]) / "hf_dataset"
    DatasetDict({"train": train, "validation": val}).save_to_disk(str(out))
    print(f"Saved dataset: train={len(train)} val={len(val)} -> {out}")


# --------------------------------------------------------------- synthetic path


def _load_techvoice(cfg: dict):
    """Returns (train Dataset | None, eval rows for asr_eval.jsonl).

    TechVoice metadata rows carry an audio_filepath relative to the dataset
    repo (no embedded audio column), so the wav files are fetched with
    snapshot_download and decoded locally.
    """
    tv_cfg = cfg["data"].get("techvoice", {})
    if not tv_cfg.get("enabled"):
        return None, []
    import soundfile as sf
    from datasets import Dataset, load_dataset
    from huggingface_hub import snapshot_download

    sr = cfg["data"]["sample_rate"]
    ds = load_dataset(tv_cfg["dataset"], split="train")
    text_col = next(c for c in ("text", "sentence", "transcript") if c in ds.column_names)
    repo_dir = Path(
        snapshot_download(tv_cfg["dataset"], repo_type="dataset", allow_patterns=["audio/*"])
    )
    ds = ds.shuffle(seed=42)

    n_train = int(len(ds) * tv_cfg.get("train_fraction", 0.7))

    def gen():
        for ex in ds.select(range(n_train)):
            yield {
                "audio": {
                    "array": _read_audio(str(repo_dir / ex["audio_filepath"]), sr),
                    "sampling_rate": sr,
                },
                "transcript": ex[text_col],
            }

    train = Dataset.from_generator(gen, features=_audio_features())

    # Held-out portion -> local wavs + eval rows (real unseen-speaker test set).
    heldout_dir = ROOT / cfg["data"]["processed_dir"] / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = []
    for i in range(n_train, len(ds)):
        ex = ds[i]
        path = heldout_dir / f"techvoice_{i:04d}.wav"
        if not path.exists():
            sf.write(path, _read_audio(str(repo_dir / ex["audio_filepath"]), sr), sr)
        eval_rows.append(
            {
                "audio_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "transcript": ex[text_col],
                "source": "techvoice",
            }
        )
    print(f"TechVoice: {len(train)} train, {len(eval_rows)} held out for eval.")
    return train, eval_rows


def build_synthetic(cfg: dict) -> None:
    from datasets import Dataset, concatenate_datasets

    syn_dir = ROOT / cfg["data"]["synthetic"]["out_dir"]
    manifest_path = syn_dir / "synth_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing — run gen_sentences then synthesize first")
    manifest = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    sr = cfg["data"]["sample_rate"]

    def to_dataset(rows: list[dict]) -> Dataset:
        def gen():
            for r in rows:
                yield {
                    "audio": {
                        "array": _read_audio(str(ROOT / r["audio_path"]), sr),
                        "sampling_rate": sr,
                    },
                    "transcript": r["text"],
                }

        return Dataset.from_generator(gen, features=_audio_features())

    synth_train = to_dataset([r for r in manifest if r["split"] == "train"])
    synth_val_rows = [r for r in manifest if r["split"] == "val"]
    val = to_dataset(synth_val_rows)

    parts = [synth_train]
    tv_train, tv_eval_rows = _load_techvoice(cfg)
    if tv_train is not None:
        parts.append(tv_train)

    cv = _load_common_voice(cfg, len(synth_train) * cfg["data"]["common_voice"]["multiplier"])
    if cv is not None:
        parts.append(cv)

    train = concatenate_datasets(parts).shuffle(seed=42)
    _save_dataset(train, val, cfg)

    # Regenerate the ASR eval fixture: real-speaker + cross-voice rows.
    eval_rows = tv_eval_rows + [
        {"audio_path": r["audio_path"], "transcript": r["text"], "source": "synth_heldout"}
        for r in synth_val_rows
    ]
    eval_path = ROOT / "data" / "evals" / "asr_eval.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(eval_rows)} eval rows -> {eval_path}")


# ------------------------------------------------------ legacy own-voice path


def segment_audio(raw_dir: Path, out_dir: Path, max_seconds: int, sample_rate: int) -> list[Path]:
    """Split each raw recording on silence into <=max_seconds chunks, 16kHz mono wav."""
    import librosa
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    for src in sorted(raw_dir.glob("*")):
        if src.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
            continue
        audio, _ = librosa.load(src, sr=sample_rate, mono=True)
        intervals = librosa.effects.split(audio, top_db=35)
        max_samples = max_seconds * sample_rate
        cur_start, cur_end, idx = None, None, 0
        for start, end in intervals:
            if cur_start is None:
                cur_start, cur_end = start, end
            elif end - cur_start <= max_samples:
                cur_end = end
            else:
                path = out_dir / f"{src.stem}_{idx:04d}.wav"
                sf.write(path, audio[cur_start:cur_end], sample_rate)
                segments.append(path)
                idx += 1
                cur_start, cur_end = start, end
        if cur_start is not None:
            path = out_dir / f"{src.stem}_{idx:04d}.wav"
            sf.write(path, audio[cur_start:cur_end], sample_rate)
            segments.append(path)
    return segments


def bootstrap_transcripts(segments: list[Path], tsv_path: Path) -> None:
    """First-pass transcripts from base Whisper, written to a correction-friendly TSV."""
    from faster_whisper import WhisperModel

    model = WhisperModel("small", compute_type="auto")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["audio_path", "transcript", "corrected"])
        for seg in segments:
            result, _ = model.transcribe(str(seg), language="en")
            text = " ".join(s.text.strip() for s in result)
            writer.writerow([str(seg), text, 0])
            print(f"{seg.name}: {text[:80]}")
    print(f"\nWrote {tsv_path}. Hand-correct the transcript column, then rerun with --from-tsv.")


def build_from_tsv(cfg: dict) -> None:
    """Corrected TSV (+ Common Voice mix-in) -> DatasetDict. Legacy own-voice path."""
    from datasets import Audio, Dataset, concatenate_datasets

    data_cfg = cfg["data"]
    tsv_path = Path(data_cfg["transcripts_tsv"])
    rows = list(csv.DictReader(open(tsv_path, encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise SystemExit(f"No rows in {tsv_path}")
    uncorrected = sum(1 for r in rows if r.get("corrected") == "0")
    if uncorrected:
        print(f"WARNING: {uncorrected}/{len(rows)} transcripts still marked uncorrected.")

    personal = Dataset.from_dict(
        {"audio": [r["audio_path"] for r in rows], "transcript": [r["transcript"] for r in rows]}
    ).cast_column("audio", Audio(sampling_rate=data_cfg["sample_rate"]))

    vocab = [t.lower() for t in data_cfg.get("custom_vocab", [])]
    split = personal.train_test_split(test_size=data_cfg.get("val_fraction", 0.1), seed=42)
    train, val = split["train"], split["test"]
    val_text = " ".join(val["transcript"]).lower()
    missing = [t for t in vocab if t not in val_text]
    if missing:
        print(f"NOTE: custom vocab missing from val split: {missing} — record more examples.")

    cv = _load_common_voice(cfg, len(train) * data_cfg["common_voice"]["multiplier"])
    if cv is not None:
        train = concatenate_datasets([train, cv]).shuffle(seed=42)
    _save_dataset(train, val, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr_finetune.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="build from synthetic renders + TechVoice + Common Voice")
    parser.add_argument("--from-tsv", action="store_true",
                        help="legacy: build from the hand-corrected TSV")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.synthetic:
        build_synthetic(cfg)
    elif args.from_tsv:
        build_from_tsv(cfg)
    else:
        segs = segment_audio(
            Path(cfg["data"]["raw_dir"]),
            Path(cfg["data"]["processed_dir"]) / "segments",
            cfg["data"]["max_segment_seconds"],
            cfg["data"]["sample_rate"],
        )
        print(f"Segmented into {len(segs)} clips.")
        bootstrap_transcripts(segs, Path(cfg["data"]["transcripts_tsv"]))


if __name__ == "__main__":
    main()
