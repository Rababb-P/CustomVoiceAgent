---
source: projects/voice-agent
topic: projects
last_updated: 2026-07-05
---

# Voice Persona Agent (this project)

## What it is

A voice-to-voice AI that answers questions about me, as me — the very system
you're talking to right now. You speak into a mic, a fine-tuned Whisper model
transcribes it, a LangGraph agent retrieves facts about my life from a personal
corpus with RAG, layered guardrails block prompt injection and hallucinated
claims, and a fully local TTS engine speaks the answer back. Everything runs
locally except the LLM, which rides the Gemini free tier behind a rate-limited,
disk-cached client I wrote — so the total API cost is zero dollars.

## Custom speech recognition

Base Whisper mangles my domain vocabulary — words like WATonomous, Reparo, and
YOLOv11 — which poisons retrieval before the agent even starts. No public
dataset contains those words, so I generate one: LLM-written sentences using
the vocabulary, rendered by a dozen different local TTS voices with speed and
noise augmentation, mixed with real human tech speech so it generalizes past
TTS artifacts. Then whisper-small is fine-tuned with LoRA adapters and exported
to CTranslate2 for real-time inference. The eval is honest: validation uses
held-out sentences spoken by held-out voices, and the fine-tune has to beat
base Whisper plus hotword biasing to justify existing.

## The agent can't hallucinate my life

It's not a stuffed prompt — it's a LangGraph state machine where retrieval is a
tool call and the guards are graph nodes. Every factual claim in an answer has
to be supported by chunks retrieved that turn; a judge model checks this, gives
the model one regeneration attempt with feedback, then falls back to an honest
"not sure". A PII denylist hard-blocks phone numbers, addresses, IDs, and
anything about third parties, no matter what's asked.

## Evals and engineering

Every change is gated by four eval suites — speech recognition word error rate,
retrieval recall, LLM-as-judge answer quality, and a 30-case red-team suite —
with a regression gate that fails CI if quality drops. Latency is a feature
too: the token stream is cut at sentence boundaries and each sentence is
synthesized while the LLM is still generating, so the first audio plays early.
The stack is Python, LangGraph, ChromaDB, faster-whisper, PEFT/LoRA, Kokoro
TTS, silero VAD, and FastAPI with WebSockets.
