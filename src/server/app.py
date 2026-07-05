"""FastAPI voice server.

WS /converse — client streams 16kHz mono PCM16 mic audio up; server detects end
of speech (VAD), transcribes, runs the guarded agent with token streaming, cuts
sentences, synthesizes locally, and streams PCM16 audio down. JSON events
(transcript, answer_chunk, sources, guard, timings, error) ride the same socket
so the client can render captions.

Streaming note: the groundedness judge needs the full answer, so on the
streaming path each sentence gets the fast local PII scan before synthesis and
the groundedness check runs on POST /ask and in evals. Latency wins; the PII
denylist still blocks hard before anything is spoken.

POST /ask — audio file in, JSON (+ base64 wav) out. Used by tests and evals.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import wave
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.config import ROOT, load_config
from src.guardrails.policy import find_pii
from src.tts.chunker import chunk_stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-load everything once: TTS model, ASR model, compiled graph.
    cfg = load_config("agent")
    state["cfg"] = cfg

    from src.agent.graph import build_graph

    state["graph"] = build_graph()
    logger.info("agent graph compiled")

    from src.asr.transcribe import _load_model

    _, asr_name = _load_model()
    logger.info("ASR ready: %s", asr_name)

    try:
        from src.tts.speak import create_engine

        state["tts"] = create_engine(cfg)
        logger.info("TTS ready: %s", cfg["tts"]["engine"])
    except Exception as e:
        state["tts"] = None
        logger.warning("TTS unavailable (%s); serving text-only", e)

    yield
    state.clear()


app = FastAPI(title="Voice Persona Agent v2", lifespan=lifespan)


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def _speak_sentences(ws: WebSocket, sentences: list[str], timings: dict) -> None:
    """Synthesize each sentence and stream PCM16 frames; PII gate per sentence."""
    tts = state["tts"]
    first_audio_at = None
    for sentence in sentences:
        if find_pii(sentence):
            logger.error("PII detected in sentence, muting turn")
            await ws.send_text(json.dumps({"type": "error", "detail": "output blocked"}))
            return
        await ws.send_text(json.dumps({"type": "answer_chunk", "text": sentence}))
        if tts is None:
            continue
        try:
            async for frame in tts.synthesize(sentence):
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                    timings["first_audio_s"] = round(first_audio_at - timings["_t0"], 2)
                await ws.send_bytes(frame)
        except Exception as e:
            # TTS crash degrades gracefully: text already sent, tell the client.
            logger.exception("TTS failed mid-turn")
            await ws.send_text(json.dumps({"type": "error", "detail": f"tts: {e}"}))
            return


async def _run_turn(ws: WebSocket, audio: bytes, thread_id: str) -> None:
    from src.agent.graph import ask
    from src.asr.transcribe import transcribe

    timings: dict = {"_t0": time.perf_counter()}
    t0 = timings["_t0"]

    wav = _pcm_to_wav_bytes(audio, 16000)
    result = await asyncio.to_thread(transcribe, wav)
    timings["asr_s"] = round(time.perf_counter() - t0, 2)
    await ws.send_text(json.dumps({"type": "transcript", "text": result.text}))
    if not result.text.strip():
        await ws.send_text(json.dumps({"type": "turn_end", "timings": {}}))
        return

    turn = await asyncio.to_thread(ask, state["graph"], result.text, thread_id=thread_id)
    timings["agent_s"] = round(time.perf_counter() - t0 - timings["asr_s"], 2)

    sentences = list(chunk_stream([turn["answer"]]))
    await _speak_sentences(ws, sentences, timings)

    timings["total_s"] = round(time.perf_counter() - t0, 2)
    timings.pop("_t0", None)
    logger.info("turn timings: %s", timings)
    await ws.send_text(
        json.dumps(
            {
                "type": "turn_end",
                "answer_text": turn["answer"],
                "guard_decisions": turn["guard"],
                "sources": turn["chunks"][:3],
                "timings": timings,
            }
        )
    )


@app.websocket("/converse")
async def converse(ws: WebSocket) -> None:
    from src.server.vad import EndOfSpeechDetector

    await ws.accept()
    cfg = state["cfg"]
    sr = state["tts"].sample_rate if state["tts"] else 24000
    await ws.send_text(json.dumps({"type": "config", "tts_sample_rate": sr}))

    vad = EndOfSpeechDetector(silence_ms=cfg["server"]["vad_silence_ms"])
    buffer = bytearray()
    thread_id = f"ws-{id(ws)}"
    try:
        while True:
            message = await ws.receive()
            if message.get("bytes"):
                buffer.extend(message["bytes"])
                if vad.update(message["bytes"]):
                    audio, _ = bytes(buffer), buffer.clear()
                    vad.reset()
                    await _run_turn(ws, audio, thread_id)
            elif message.get("text"):
                event = json.loads(message["text"])
                if event.get("type") == "end_turn":  # client-side push-to-talk release
                    if buffer:
                        audio, _ = bytes(buffer), buffer.clear()
                        vad.reset()
                        await _run_turn(ws, audio, thread_id)
            elif message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass


@app.post("/ask")
async def ask_endpoint(file: UploadFile):
    """Audio in -> JSON out (transcript, answer, sources, guard, base64 wav).
    Non-streaming path: full guard stack including groundedness runs here."""
    from src.agent.graph import ask
    from src.asr.transcribe import transcribe

    audio = await file.read()
    result = await asyncio.to_thread(transcribe, audio)
    turn = await asyncio.to_thread(ask, state["graph"], result.text, thread_id="http")

    audio_b64 = None
    if state["tts"] is not None and not find_pii(turn["answer"]):
        pcm = bytearray()
        try:
            async for frame in state["tts"].synthesize(turn["answer"]):
                pcm.extend(frame)
            audio_b64 = base64.b64encode(
                _pcm_to_wav_bytes(bytes(pcm), state["tts"].sample_rate)
            ).decode()
        except Exception:
            logger.exception("TTS failed on /ask")

    return {
        "transcript": result.text,
        "answer": turn["answer"],
        "guard_decisions": turn["guard"],
        "sources": turn["chunks"][:3],
        "audio_wav_base64": audio_b64,
    }


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "static" / "index.html").read_text(encoding="utf-8"))
