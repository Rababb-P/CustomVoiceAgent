"""End-of-speech detection over 16kHz mono PCM16 frames.

Silero VAD when available; a simple RMS energy gate as fallback so the server
runs without the tts extra installed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class EndOfSpeechDetector:
    """Feed PCM16 frames; `update()` returns True once speech has occurred and
    was followed by `silence_ms` of silence."""

    def __init__(self, silence_ms: int = 300):
        self._silence_samples = int(SAMPLE_RATE * silence_ms / 1000)
        self._silent_run = 0
        self._heard_speech = False
        self._silero = None
        try:
            from silero_vad import load_silero_vad

            self._silero = load_silero_vad()
            self._buf = np.zeros(0, dtype=np.float32)
        except Exception:
            logger.info("silero-vad unavailable; using RMS energy gate")

    def _is_speech(self, frame_f32: np.ndarray) -> bool:
        if self._silero is not None:
            import torch

            # Silero consumes fixed 512-sample windows at 16kHz.
            self._buf = np.concatenate([self._buf, frame_f32])
            speech = False
            while len(self._buf) >= 512:
                window, self._buf = self._buf[:512], self._buf[512:]
                prob = self._silero(torch.from_numpy(window), SAMPLE_RATE).item()
                speech = speech or prob > 0.5
            return speech
        return float(np.sqrt(np.mean(frame_f32**2))) > 0.015

    def update(self, pcm16: bytes) -> bool:
        frame = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if self._is_speech(frame):
            self._heard_speech = True
            self._silent_run = 0
        else:
            self._silent_run += len(frame)
        return self._heard_speech and self._silent_run >= self._silence_samples

    def reset(self) -> None:
        self._silent_run = 0
        self._heard_speech = False
        if self._silero is not None:
            self._buf = np.zeros(0, dtype=np.float32)
