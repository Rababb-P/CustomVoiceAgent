"""Transcription via faster-whisper: CLI + importable function.

Loads the fine-tuned CT2 export if present, else falls back to the base model
named in configs/agent.yaml (server.asr_fallback) so the rest of the stack works
before Phase 1 is complete.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass, field
from pathlib import Path

from src.config import load_config


@dataclass
class Transcription:
    text: str
    segments: list[dict] = field(default_factory=list)  # {text, start, end, confidence}
    model_name: str = ""


@functools.lru_cache(maxsize=1)
def _load_model():
    from faster_whisper import WhisperModel

    cfg = load_config("agent")["server"]
    model_dir = Path(cfg["asr_model_dir"])
    if model_dir.exists():
        return WhisperModel(str(model_dir), compute_type="auto"), str(model_dir)
    return WhisperModel(cfg["asr_fallback"], compute_type="auto"), cfg["asr_fallback"]


@functools.lru_cache(maxsize=1)
def _hotwords() -> str | None:
    """Custom-vocab decoding bias — free accuracy on rare terms (WATonomous,
    Reparo, ...) for any speaker, with or without the fine-tuned model."""
    if not load_config("agent")["server"].get("hotwords_from_vocab", False):
        return None
    vocab = load_config("asr_finetune")["data"].get("custom_vocab", [])
    return " ".join(vocab) or None


def transcribe(audio: str | Path | bytes) -> Transcription:
    """Transcribe a file path or raw 16kHz mono PCM/wav bytes."""
    import io

    model, name = _load_model()
    source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
    segments, _info = model.transcribe(
        source, language="en", vad_filter=True, hotwords=_hotwords()
    )
    segs = [
        {
            "text": s.text.strip(),
            "start": s.start,
            "end": s.end,
            # avg_logprob is a log-probability; exp() gives a rough 0-1 confidence
            "confidence": round(2.718281828 ** s.avg_logprob, 3),
        }
        for s in segments
    ]
    return Transcription(
        text=" ".join(s["text"] for s in segs), segments=segs, model_name=name
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="path to an audio file")
    args = parser.parse_args()
    result = transcribe(args.audio)
    print(f"[{result.model_name}]")
    for seg in result.segments:
        print(f"  {seg['start']:6.1f}-{seg['end']:6.1f}  ({seg['confidence']:.2f})  {seg['text']}")
    print(f"\n{result.text}")


if __name__ == "__main__":
    main()
