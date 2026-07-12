"""Compare base and exported adapters on synthetic validation and real test speech.

python -m scripts.evaluate_asr_retrain --base-only
python -m scripts.evaluate_asr_retrain
"""

import argparse
import json

from src.asr.prepare_data import _read_audio
from src.config import ROOT, load_config


def real_test_rows(cfg, count=32):
    import soundfile as sf
    from datasets import Audio, load_dataset

    destination = ROOT / cfg["data"]["processed_dir"] / "real_test"
    destination.mkdir(parents=True, exist_ok=True)
    index = destination / "manifest.jsonl"
    if index.exists():
        return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    cv = cfg["data"]["common_voice"]
    data = load_dataset(cv["dataset"], cv["language"], split="test", streaming=True)
    data = data.cast_column("audio", Audio(decode=False))
    rows = []
    for i, example in enumerate(data.take(count)):
        path = destination / f"{i:03d}.wav"
        sf.write(path, _read_audio(example["audio"]["bytes"], 16000), 16000)
        rows.append(
            {
                "audio_path": path.relative_to(ROOT).as_posix(),
                "text": example["sentence"],
                "source": "common_voice_test",
            }
        )
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def score(model_dir, rows, vocab, hotwords=None):
    import jiwer
    from faster_whisper import WhisperModel

    from evals.run_asr_eval import _term_accuracy

    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=4)
    predictions = []
    for i, row in enumerate(rows):
        segments, _ = model.transcribe(
            str(ROOT / row["audio_path"]),
            language="en",
            beam_size=1,
            temperature=0,
            condition_on_previous_text=False,
            hotwords=hotwords,
        )
        predictions.append(" ".join(segment.text.strip() for segment in segments))
        if (i + 1) % 16 == 0:
            print(f"Decoded {i + 1}/{len(rows)} clips from {model_dir}", flush=True)
    references = [row["text"] for row in rows]
    normalize = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    return {
        "n": len(rows),
        "wer": jiwer.wer(references, predictions),
        "normalized_wer": jiwer.wer(normalize(references), normalize(predictions)),
        "cer": jiwer.cer(references, predictions),
        "term_accuracy": _term_accuracy(references, predictions, vocab),
        "predictions": [
            {"reference": reference, "prediction": prediction}
            for reference, prediction in zip(references, predictions, strict=True)
        ],
    }


def main():
    from ctranslate2.converters import TransformersConverter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config("asr_retrain_cpu")
    real = real_test_rows(cfg)
    manifest = ROOT / cfg["data"]["synthetic"]["out_dir"] / "synth_manifest.jsonl"
    synthetic = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == "val"
    ]
    groups = {"synthetic_validation": synthetic, "common_voice_test": real}
    # Keep make eval-asr pointed at the audio used for this recovery run.
    fixture = ROOT / "data/evals/asr_eval.jsonl"
    fixture.write_text(
        "".join(
            json.dumps({"audio_path": row["audio_path"], "transcript": row["text"], "source": name})
            + "\n"
            for name, rows in groups.items()
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    out = ROOT / cfg["training"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    base_dir = ROOT / "models/whisper-small-base-ct2"
    if not (base_dir / "model.bin").exists():
        TransformersConverter(
            cfg["model"]["base"], copy_files=["tokenizer.json", "preprocessor_config.json"]
        ).convert(str(base_dir), quantization="int8")
    base_results = out / "base_evaluation.json"
    if base_results.exists():
        results = json.loads(base_results.read_text(encoding="utf-8"))
    else:
        results = {
            "decoding": {
                "beam_size": 1,
                "temperature": 0,
                "language": "en",
                "condition_on_previous_text": False,
                "compute_type": "int8",
            },
            "base": {},
            "base_hotwords": {},
        }
        for name, rows in groups.items():
            results["base"][name] = score(base_dir, rows, cfg["data"]["custom_vocab"])
            results["base_hotwords"][name] = score(
                base_dir,
                rows,
                cfg["data"]["custom_vocab"],
                hotwords=" ".join(cfg["data"]["custom_vocab"]),
            )
        base_results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not args.base_only:
        results["finetuned"] = {}
        for name, rows in groups.items():
            results["finetuned"][name] = score(
                ROOT / cfg["export"]["ct2_dir"], rows, cfg["data"]["custom_vocab"]
            )
        (out / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for variant in ("base", "base_hotwords", "finetuned"):
        for group, values in results.get(variant, {}).items():
            print(
                f"{variant}/{group}: WER={values['wer']:.2%}, "
                f"normalized WER={values['normalized_wer']:.2%}"
            )


if __name__ == "__main__":
    main()
