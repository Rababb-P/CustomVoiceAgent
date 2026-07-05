"""Local TTS behind a common async interface.

Primary: Chatterbox (Resemble AI, MIT) — zero-shot voice cloning from a 10-20s
reference clip. Fallback: Kokoro — much faster, tiny, but generic voice.
Both are warm-loaded once (at server start) and expose cancel() so barge-in can
cut a turn short. `python -m src.tts.speak --bench` compares synthesis speed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import ROOT, load_config

logger = logging.getLogger(__name__)


class TTSEngine(Protocol):
    sample_rate: int

    async def synthesize(self, text: str) -> AsyncIterator[bytes]: ...
    def cancel(self) -> None: ...


class _BaseEngine:
    sample_rate = 24000

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _reset(self) -> None:
        self._cancelled = False

    @staticmethod
    def _to_pcm16(wav: np.ndarray) -> bytes:
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767).astype(np.int16).tobytes()


class ChatterboxEngine(_BaseEngine):
    """Voice-cloned synthesis. GPU strongly recommended; CPU works with a warning."""

    def __init__(self, reference_clip: Path):
        super().__init__()
        import torch
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("Chatterbox on CPU — expect multi-second synthesis. "
                           "Consider tts.engine: kokoro in configs/agent.yaml.")
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = self._model.sr
        if not reference_clip.exists():
            raise FileNotFoundError(
                f"Voice reference clip missing: {reference_clip}. "
                "Pick a clean 10-20s clip from the Phase 1 recordings."
            )
        self._ref = str(reference_clip)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self._reset()
        loop = asyncio.get_running_loop()
        wav = await loop.run_in_executor(
            None, lambda: self._model.generate(text, audio_prompt_path=self._ref)
        )
        if self._cancelled:
            return
        yield self._to_pcm16(wav.squeeze().cpu().numpy())


class KokoroEngine(_BaseEngine):
    """Fast generic-voice fallback; fine on CPU."""

    def __init__(self, voice: str = "am_adam"):
        super().__init__()
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code="a")  # American English
        self._voice = voice
        self.sample_rate = 24000

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self._reset()
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(
            None, lambda: list(self._pipeline(text, voice=self._voice))
        )
        for _, _, audio in chunks:
            if self._cancelled:
                return
            yield self._to_pcm16(np.asarray(audio))


def create_engine(config: dict | None = None) -> TTSEngine:
    cfg = (config or load_config("agent"))["tts"]
    if cfg["engine"] == "chatterbox":
        return ChatterboxEngine(ROOT / cfg["reference_clip"])
    if cfg["engine"] == "kokoro":
        return KokoroEngine(cfg.get("kokoro_voice", "am_adam"))
    raise ValueError(f"Unknown TTS engine: {cfg['engine']}")


# ------------------------------------------------------------------ benchmark

_BENCH_TEXT = (
    "Hey, I'm Rababb. I study engineering at Waterloo and spend most of my time "
    "on machine learning projects, autonomous vehicles, and the occasional hackathon."
)


async def _bench_one(name: str, engine: TTSEngine) -> None:
    start = time.perf_counter()
    first = None
    total_bytes = 0
    async for chunk in engine.synthesize(_BENCH_TEXT):
        first = first or time.perf_counter() - start
        total_bytes += len(chunk)
    elapsed = time.perf_counter() - start
    audio_secs = total_bytes / 2 / engine.sample_rate
    print(
        f"{name:12s} first audio {first:6.2f}s | total {elapsed:6.2f}s | "
        f"{audio_secs:5.1f}s audio | RTF {elapsed / max(audio_secs, 0.01):.2f}"
    )


def bench() -> None:
    cfg = load_config("agent")
    for name in ("chatterbox", "kokoro"):
        try:
            engine = create_engine({**cfg, "tts": {**cfg["tts"], "engine": name}})
            asyncio.run(_bench_one(name, engine))
        except Exception as e:
            print(f"{name:12s} skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if args.bench:
        bench()
