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
import copy
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
from src.tts.chunker import achunk_stream, chunk_stream

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
    # Streaming graph for the voice path. The groundedness judge needs the
    # full answer, so per the design it is skipped here (the local PII gate
    # runs per sentence instead) — which also means tokens can be spoken as
    # they stream and no regeneration can contradict already-spoken audio.
    ws_cfg = copy.deepcopy(cfg)
    ws_cfg["guards"]["output"]["groundedness_check"] = False
    ws_cfg["guards"]["output"]["max_regenerations"] = 0
    state["graph_ws"] = build_graph(judge=None, config=ws_cfg)
    logger.info("agent graphs compiled (full + streaming)")

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


async def _drain(queue: asyncio.Queue) -> None:
    """Consume a sentence queue to its None sentinel so the producer never
    blocks on put() after the speaking side bails out early."""
    while await queue.get() is not None:
        pass


async def _speak_sentences(ws: WebSocket, sentences: asyncio.Queue, timings: dict) -> None:
    """Synthesize each queued sentence and stream PCM16 frames; PII gate per
    sentence. Consumes until the None sentinel."""
    tts = state["tts"]
    first_audio_at = None
    while (sentence := await sentences.get()) is not None:
        if find_pii(sentence):
            logger.error("PII detected in sentence, muting turn")
            await ws.send_text(json.dumps({"type": "error", "detail": "output blocked"}))
            await _drain(sentences)
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
            await _drain(sentences)
            return


async def _run_turn(ws: WebSocket, audio: bytes, thread_id: str) -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphRecursionError

    from src.agent.graph import SAFE_FALLBACK, _text, recursion_limit
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

    meta: dict = {"guard": {}, "chunks": [], "fallback": ""}
    parts: list[str] = []

    async def tokens():
        """Final-answer tokens from the streaming graph; guard verdicts,
        retrieved chunks, and non-streamed messages (refusals, clarify) are
        collected into meta on the side."""
        async for mode, payload in state["graph_ws"].astream(
            {"messages": [HumanMessage(content=result.text)]},
            config={"configurable": {"thread_id": thread_id},
                    "recursion_limit": recursion_limit(state["cfg"])},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, md = payload
                if md.get("langgraph_node") != "agent" or getattr(chunk, "tool_calls", None) \
                        or getattr(chunk, "tool_call_chunks", None):
                    continue
                text = _text(chunk)
                if text:
                    parts.append(text)
                    yield text
            else:
                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue
                    meta["guard"] = update.get("guard") or meta["guard"]
                    meta["chunks"] = update.get("chunks") or meta["chunks"]
                    guard_nodes = ("input_guard", "clarify", "output_guard")
                    if node in guard_nodes and update.get("messages"):
                        meta["fallback"] = _text(update["messages"][-1])

    # The queue decouples token production from synthesis: the LLM keeps
    # generating while a sentence is being spoken.
    queue: asyncio.Queue = asyncio.Queue()

    async def produce() -> None:
        spoken = 0
        try:
            async for sentence in achunk_stream(tokens()):
                await queue.put(sentence)
                spoken += 1
        except GraphRecursionError:
            logger.warning("recursion limit hit; speaking safe fallback")
            meta["fallback"] = SAFE_FALLBACK
        finally:
            timings["agent_s"] = round(time.perf_counter() - t0 - timings["asr_s"], 2)
            if not spoken and meta["fallback"]:
                # Guard refusal / clarify / fallback: nothing streamed, speak it.
                for s in chunk_stream([meta["fallback"]]):
                    await queue.put(s)
            await queue.put(None)

    producer = asyncio.create_task(produce())
    await _speak_sentences(ws, queue, timings)
    await producer  # surface graph errors after the queue is drained

    timings["total_s"] = round(time.perf_counter() - t0, 2)
    timings.pop("_t0", None)
    logger.info("turn timings: %s", timings)
    await ws.send_text(
        json.dumps(
            {
                "type": "turn_end",
                "answer_text": "".join(parts).strip() or meta["fallback"],
                "guard_decisions": meta["guard"],
                "sources": meta["chunks"][:3],
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
    # The turn runs as a task so this loop keeps consuming the socket: uvicorn
    # applies read backpressure when the app stops receiving, which also stops
    # ping/pong processing — the keepalive then kills the connection (1011)
    # mid-turn. Mic frames that arrive while a turn is running are discarded.
    turn: asyncio.Task | None = None

    def start_turn(audio: bytes) -> asyncio.Task:
        return asyncio.create_task(_run_turn(ws, audio, thread_id))

    try:
        while True:
            message = await ws.receive()
            if turn and turn.done():
                turn.result()  # surface exceptions from the finished turn
                turn = None
            if message.get("bytes"):
                if turn:
                    continue
                buffer.extend(message["bytes"])
                if vad.update(message["bytes"]):
                    audio, _ = bytes(buffer), buffer.clear()
                    vad.reset()
                    turn = start_turn(audio)
            elif message.get("text"):
                event = json.loads(message["text"])
                if event.get("type") == "end_turn":  # client-side push-to-talk release
                    if buffer and not turn:
                        audio, _ = bytes(buffer), buffer.clear()
                        vad.reset()
                        turn = start_turn(audio)
            elif message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if turn and not turn.done():
            turn.cancel()


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
