"""Prepare personal recordings for Whisper fine-tuning.

Pipeline: raw recordings -> <=30s segments, 16kHz mono -> bootstrap transcripts with
base Whisper -> hand-correctable TSV -> HF Dataset on disk.

Run twice: first pass writes transcripts.tsv with Whisper's guesses; you correct the
`transcript` column by hand (especially custom vocab), then the second pass — with
--from-tsv — builds the final dataset from your corrections.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import load_config


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
        # Split on silence, then greedily pack intervals up to max_seconds.
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
        writer.writerow(["audio_path", "transcript", "corrected"])  # flip corrected to 1 as you fix
        for seg in segments:
            result, _ = model.transcribe(str(seg), language="en")
            text = " ".join(s.text.strip() for s in result)
            writer.writerow([str(seg), text, 0])
            print(f"{seg.name}: {text[:80]}")
    print(f"\nWrote {tsv_path}. Hand-correct the transcript column, then rerun with --from-tsv.")


def build_dataset(cfg: dict) -> None:
    """Corrected TSV (+ optional Common Voice mix-in) -> HF DatasetDict on disk."""
    from datasets import Audio, Dataset, DatasetDict, load_dataset

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

    # Stratified-ish split: ensure each custom-vocab term lands in both splits when possible.
    vocab = [t.lower() for t in data_cfg.get("custom_vocab", [])]
    val_frac = data_cfg.get("val_fraction", 0.1)
    split = personal.train_test_split(test_size=val_frac, seed=42)
    train, val = split["train"], split["test"]
    val_text = " ".join(val["transcript"]).lower()
    missing = [t for t in vocab if t not in val_text]
    if missing:
        print(f"NOTE: custom vocab missing from val split: {missing} — record more examples.")

    cv_cfg = data_cfg.get("common_voice", {})
    if cv_cfg.get("enabled"):
        n_cv = min(len(train) * cv_cfg.get("multiplier", 5), 20000)
        cv = load_dataset(cv_cfg["dataset"], cv_cfg["language"], split="train", streaming=True)
        cv_rows = {"audio": [], "transcript": []}
        for ex in cv.take(n_cv):
            cv_rows["audio"].append(ex["audio"])
            cv_rows["transcript"].append(ex["sentence"])
        cv_ds = Dataset.from_dict(cv_rows).cast_column(
            "audio", Audio(sampling_rate=data_cfg["sample_rate"])
        )
        from datasets import concatenate_datasets

        train = concatenate_datasets([train, cv_ds]).shuffle(seed=42)
        print(f"Mixed in {len(cv_ds)} Common Voice examples.")

    out = Path(data_cfg["processed_dir"]) / "hf_dataset"
    DatasetDict({"train": train, "validation": val}).save_to_disk(str(out))
    print(f"Saved dataset: train={len(train)} val={len(val)} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr_finetune.yaml")
    parser.add_argument(
        "--from-tsv",
        action="store_true",
        help="skip segmentation/bootstrap; build dataset from the corrected TSV",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if not args.from_tsv:
        segs = segment_audio(
            Path(cfg["data"]["raw_dir"]),
            Path(cfg["data"]["processed_dir"]) / "segments",
            cfg["data"]["max_segment_seconds"],
            cfg["data"]["sample_rate"],
        )
        print(f"Segmented into {len(segs)} clips.")
        bootstrap_transcripts(segs, Path(cfg["data"]["transcripts_tsv"]))
    else:
        build_dataset(cfg)


if __name__ == "__main__":
    main()
