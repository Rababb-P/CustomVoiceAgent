# Voice Persona Agent v2 — Project Plan & Claude Code Instructions

Custom ASR (fine-tuned Whisper) + RAG persona agent grounded on my life/resume data, with a LangGraph agentic loop, guardrails, local voice-cloned TTS for natural back-and-forth conversation, and an eval harness that gates every change. Total API cost: $0.

**Local vs API:** ASR (fine-tuned Whisper via faster-whisper), embeddings (bge), and TTS (Chatterbox, cloned to my voice) all run fully local. The only cloud dependency is the Gemini API free tier (no credit card). Free-tier caveats to design around: ~10 RPM / ~250 requests per day on `gemini-2.5-flash` (more on flash-lite), and free-tier prompts may be used by Google for training — acceptable since the corpus is my public persona info, but never put anything private in it.

---

## 0. How to work on this project (read first, Claude Code)

- Work phase by phase. Do not start a phase until the previous phase's Definition of Done passes.
- Before writing code for a phase, output a short plan (files to create, key decisions) and wait for approval.
- Every phase ends with: tests passing, eval numbers printed, a short summary of what changed.
- Small commits, one logical change each. Conventional commit messages.
- Never hardcode API keys. Use `.env` + `python-dotenv`. Add `.env` to `.gitignore` immediately.
- Never commit audio data or the personal corpus. `data/` is gitignored except for `data/README.md` and eval JSONL fixtures with synthetic content.
- Use LangChain for RAG plumbing (loaders, splitters, Chroma vectorstore, embeddings) and LangGraph for the agent. `langchain-google-genai` for the Gemini binding. Keep graph nodes small and individually testable; no giant chains.
- All Gemini calls go through one shared client wrapper (`src/llm.py`): client-side RPM limiter, exponential backoff on 429s, and an on-disk response cache keyed by (model, prompt hash). Free tier quotas are tight; this wrapper is non-negotiable and gets built in Phase 0 scaffolding.
- Model policy: `gemini-2.5-flash` for the agent, `gemini-2.5-flash-lite` for guards and judges (higher RPM/RPD, good enough for classification).
- Prefer boring, debuggable code. Where LangChain abstractions fight us, wrap them behind our own thin interfaces (`store.py`, `retrieve.py`) so they stay swappable.
- Python 3.11, `uv` for dependency management, `ruff` for lint/format, `pytest` for tests.

## 1. Repo layout

```
voice-agent-v2/
├── pyproject.toml
├── Makefile                  # make train-asr, make eval, make serve, make redteam
├── .env.example
├── configs/
│   ├── asr_finetune.yaml
│   ├── rag.yaml
│   └── agent.yaml
├── data/                     # gitignored (except README + fixtures)
│   ├── audio/raw/            # my voice recordings
│   ├── audio/processed/
│   ├── corpus/               # resume, project writeups, personal FAQ (markdown)
│   └── evals/                # asr_eval.jsonl, rag_qa.jsonl, adversarial.jsonl
├── src/
│   ├── llm.py                # shared Gemini client: rate limiter, backoff, cache
│   ├── asr/
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   ├── export.py         # HF -> CTranslate2 for faster-whisper
│   │   └── transcribe.py
│   ├── rag/
│   │   ├── ingest.py
│   │   ├── retrieve.py
│   │   └── store.py          # Chroma wrapper
│   ├── agent/
│   │   ├── graph.py          # LangGraph state graph (guards + agent + tools)
│   │   ├── tools.py
│   │   ├── prompts.py
│   │   └── persona.py
│   ├── guardrails/
│   │   ├── input_guard.py
│   │   ├── output_guard.py
│   │   └── policy.py
│   ├── tts/
│   │   ├── speak.py          # local Chatterbox TTS, cloned to my voice
│   │   └── chunker.py        # sentence-level chunking of streamed tokens
│   └── server/
│       └── app.py            # FastAPI + WebSocket: audio in -> audio out
├── evals/
│   ├── run_asr_eval.py
│   ├── run_rag_eval.py
│   ├── run_agent_eval.py     # LLM-as-judge
│   ├── run_redteam.py
│   └── report.py             # aggregates into evals/results/latest.json
└── tests/
```

---

## Phase 1 — ASR fine-tuning (PyTorch + Hugging Face)

**Goal:** Whisper fine-tuned on my voice, beating the base model's WER on a held-out set of my own speech, exported for fast inference.

### Data
- I will record 60–120 min of my own speech: reading resume/project descriptions, ad-libbed Q&A answers, technical vocab I actually use (WATonomous, Reparo, YOLOv11, ROS2, Slurm, Thevenin, etc.).
- `prepare_data.py`: chunk to ≤30s segments, resample 16kHz mono, build a HF `Dataset` with (audio, transcript) pairs. Transcripts bootstrapped with base Whisper then hand-corrected (script should output a correction-friendly TSV).
- Mix in a slice of Common Voice EN (~5–10x my data volume) so the model doesn't overfit to my voice and forget general English.
- Split: 90/10 train/val on my recordings, stratified so rare technical terms appear in both.

### Training
- Base model: `openai/whisper-small` (fits on a single consumer GPU with LoRA; config flag to swap to medium).
- PEFT LoRA on attention projections (`q_proj`, `v_proj`), r=32, alpha=64 as starting point. Full config in `configs/asr_finetune.yaml`.
- HF `Seq2SeqTrainer`, fp16, gradient accumulation, `predict_with_generate=True`, WER as the eval metric via `jiwer`.
- Log to a local `runs/` dir (tensorboard). No external tracking services.

### Export & inference
- `export.py`: merge LoRA weights, convert to CTranslate2 (`ct2-transformers-converter`) so it drops into faster-whisper — same runtime as my v1 Lambda deployment.
- `transcribe.py`: CLI + importable function, returns text plus segment-level confidence.

### Definition of Done
- `make train-asr` runs end to end from processed data.
- `make eval-asr` prints WER for base whisper-small vs fine-tuned on the held-out set. Fine-tuned must beat base, with a per-term report on my custom vocab (Reparo, WATonomous, etc.).
- Exported CT2 model transcribes a sample file via faster-whisper in `transcribe.py`.

---

## Phase 2 — RAG over life corpus

**Goal:** Retrieval layer over my personal corpus that reliably surfaces the right facts.

### Corpus
- `data/corpus/` as markdown files: resume.md, projects/reparo.md, projects/watonomous.md, work/bmo.md, education.md, faq.md (hackathons, interests, ball hockey, etc.). I'll write these; generate stubs with frontmatter (`source`, `topic`, `last_updated`).

### Pipeline
- `ingest.py`: LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` (~300 tokens, 50 overlap), each chunk tagged with source file + heading path in metadata.
- Embeddings: `BAAI/bge-small-en-v1.5` via `HuggingFaceEmbeddings` (local, free, fast). Config flag to swap models.
- Store: LangChain `Chroma` vectorstore persisted to `data/.chroma/`, wrapped in `store.py` so the vector DB is swappable.
- `retrieve.py`: retriever with top-k (default 6) and similarity threshold; returns chunks with metadata. Optional cross-encoder rerank (`bge-reranker-base` via `ContextualCompressionRetriever`) behind a config flag.

### Definition of Done
- `make ingest` builds the index; re-running is idempotent.
- `make eval-rag` runs `rag_qa.jsonl` (I'll write ~40 questions with gold source chunks) and prints recall@k and MRR. Target: recall@6 ≥ 0.9 on the seed set.

---

## Phase 3 — Agentic loop

**Goal:** A LangGraph agent that answers as "AI Rababb" using tool calls, not a single stuffed prompt.

### Design
- `langgraph` StateGraph in `graph.py`, LLM via `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) routed through the shared `llm.py` wrapper, streaming enabled end to end (needed for low-latency TTS in Phase 6).
- Graph nodes: `input_guard` -> `agent` <-> `tools` (standard tool loop, max 5 iterations via recursion limit) -> `output_guard` -> END. Guards from Phase 4 slot in as nodes; conditional edges route refusals straight to END.
- Tools (`tools.py`, LangChain `@tool`):
  - `search_life_info(query)` -> RAG retrieval
  - `list_topics()` -> corpus table of contents (helps the model decide what's knowable)
  - `clarify(question)` -> terminal tool that returns a clarifying question to the user instead of an answer
- Graph state: message history, retrieved chunks for the current turn (output guard needs them), guard decisions. Use LangGraph checkpointing (`MemorySaver`, thread_id per session) for multi-turn memory, capped at last 10 turns.
- Persona (`persona.py` + `prompts.py`): first person, my voice — direct, casual, concise. Hard rules: only claim facts backed by retrieved chunks; if not in corpus, say so plainly instead of hallucinating; answers must be spoken-style (short sentences, no markdown, no lists, under ~80 words unless asked for depth) since they go straight to TTS.

### Definition of Done
- `python -m src.agent.graph "what did you build at hack canada"` returns a grounded, in-persona spoken-style answer, with the node/tool trace visible in verbose mode.
- Questions outside the corpus ("what's your SIN") produce a clean deflection, not a fabrication.
- Multi-turn works: a follow-up like "what stack did you use for that" resolves against the previous turn.
- Unit tests use a fake chat model, assert the graph terminates, respects the recursion limit, and handles tool errors.

---

## Phase 4 — Guardrails

**Goal:** Layered checks so the agent can't be prompt-injected, won't leak beyond policy, and never asserts ungrounded facts. Guards are implemented as pure functions and wired into the LangGraph graph as nodes with conditional edges.

### Input guard (`input_guard.py`)
- Fast heuristic pass: injection patterns ("ignore previous instructions", role-swap attempts, requests to reveal the system prompt), plus length caps.
- Cheap classifier pass: single `gemini-2.5-flash-lite` call classifying {on_topic, off_topic, injection, sensitive_request} — runs only when heuristics are uncertain (keeps RPD spend down).
- Off-topic gets a polite in-persona redirect, injection gets a canned refusal. Neither hits the main agent.

### Output guard (`output_guard.py`)
- Groundedness check: every factual claim must be supported by the retrieved chunks in that turn. Implement as a `gemini-2.5-flash-lite` judge call: (answer, chunks) -> {grounded, ungrounded_claims[]}. Ungrounded -> one regeneration attempt with the violation injected as feedback, then fall back to a safe "not sure" response.
- PII policy in `policy.py`: explicit allowlist (school, roles, projects, public wins) and denylist (contact details beyond public email, addresses, anything about third parties like family/friends). Regex + judge check on the denylist.

### Definition of Done
- `make redteam` runs `adversarial.jsonl` (~30 cases: injections, PII fishing, off-topic bait, "pretend you're not Rababb"). Pass rate printed per category; must be 100% on PII denylist and injection categories.
- Guards are pure functions with unit tests; each guard logs its decision for debugging.

---

## Phase 5 — Eval harness (ties it all together)

**Goal:** One command that scores the whole system and fails loudly on regressions.

### Suites
1. **ASR**: WER/CER on held-out personal set + custom-vocab term accuracy (from Phase 1).
2. **Retrieval**: recall@k, MRR on `rag_qa.jsonl` (from Phase 2).
3. **End-to-end answer quality**: `run_agent_eval.py` runs ~40 questions through the full pipeline. LLM-as-judge (`gemini-2.5-flash`, via the cached wrapper) scores each answer 1–5 on: faithfulness to corpus, persona fidelity, TTS-friendliness, directness. Rubric lives in the eval file, not the prompt code. Judge sees gold facts, not gold answers.
4. **Safety**: red-team pass rates (from Phase 4).

### Mechanics
- All eval sets are JSONL in `data/evals/` with a schema documented in `data/README.md`.
- `make eval` runs all suites, writes `evals/results/<timestamp>.json`, prints a comparison table vs `latest.json`, then updates the pointer.
- Regression gate: `make eval-ci` exits non-zero if WER rises >5% relative, recall@6 drops below 0.85, mean judge score drops >0.3, or any safety category dips below its floor.
- Judge calls cached by (input hash, rubric hash) so reruns are cheap.
- Quota budgeting: a full uncached eval run is ~40 agent turns + ~40 judge calls + guard calls, which is a big chunk of the free tier's daily requests. `make eval` prints estimated request count up front, throttles to stay under RPM, and supports `--suite` flags to run subsets. Cache hits cost zero quota, so day-to-day runs should be mostly free.

### Definition of Done
- Fresh clone + `make eval` produces a full scored report.
- Deliberately breaking retrieval (e.g., k=1) makes `make eval-ci` fail. Demonstrate this once.

---

## Phase 6 — Voice out: local voice-cloned TTS + conversational loop

**Goal:** Full voice-to-voice turn that feels like a conversation, not a request/response form, with zero API cost. Target ≤3s from end of user speech to first audio out (local TTS is slower than a paid API; measure and tune).

### TTS (`src/tts/`)
- Primary: **Chatterbox** (Resemble AI, MIT license) running locally. Zero-shot voice cloning from a ~10–20s reference clip — reuse a clean segment from the Phase 1 ASR recordings. Supports streaming synthesis; run on GPU if available, CPU fallback with a warning.
- Fallback engine behind a config flag: **Kokoro** (very fast, tiny, great quality, but no voice cloning) for when latency matters more than it sounding like me, or for CPU-only environments.
- `speak.py`: common async interface over both engines, yields audio chunks (pcm/wav), exposes `cancel()` so a turn can be cut short. Warm-load the model at server start, never per request.
- `chunker.py`: consume the LangGraph token stream, cut at sentence boundaries (and clause boundaries for long sentences), synthesize each chunk as it completes so first audio plays while Gemini is still generating.

### Conversational server (`server/app.py`)
- FastAPI WebSocket endpoint `/converse`: client streams mic audio up, server streams TTS audio down.
- Turn pipeline: VAD-based end-of-speech detection (silero-vad, local) -> faster-whisper transcription -> graph (guards + agent) with token streaming -> chunker -> local TTS -> audio frames down the socket. Also emit JSON events (`transcript`, `answer_text`, `sources`, `guard_decisions`) alongside audio so the frontend can show captions.
- Barge-in (stretch, behind a flag): if VAD detects the user speaking mid-answer, cancel TTS synthesis and start a new turn. Easier here than with a cloud TTS since cancellation is instant and free.
- Keep a plain `POST /ask` (audio in -> JSON + audio file out) for testing and evals.

### Latency budget (measure and log per turn)
- VAD end-of-speech: ~300ms. ASR: <500ms for a 10s utterance. Gemini time-to-first-token: variable on free tier, log it. First TTS audio: depends on hardware — benchmark Chatterbox vs Kokoro on my machine in a `make bench-tts` target and record numbers in the README. Log each stage so regressions are visible.

### Definition of Done
- `make serve` + a minimal HTML test page (mic capture, WebSocket, audio playback) supports a full spoken multi-turn conversation in my cloned voice.
- Per-stage latency logged; `make bench-tts` reports synthesis speed for both engines.
- TTS engine crash mid-turn degrades gracefully (text still returned, error event emitted).

---

## Order of operations & checkpoints

1. Scaffold repo, tooling, Makefile, configs, CI-less pytest setup. **Checkpoint: I review layout.**
2. Phase 2 (RAG) before Phase 1 if no GPU is available yet — they're independent. Otherwise Phase 1 first.
3. Phases 3 -> 4 -> 5 in order (evals need the agent and guards to exist, but write eval JSONL schemas early so phases can target them).
4. Phase 6 last.

## Open decisions for me (Rababb) — ask before assuming
- GPU situation for training AND inference (local vs Colab vs rented). Affects whisper-small vs medium, batch config, and whether Chatterbox runs fast enough for real-time or Kokoro becomes the default.
- Which corpus docs exist already vs need writing.
- Whether v2 replaces the Lambda ASR endpoint or runs alongside it for A/B.
- Which reference clip to use for the Chatterbox voice clone (pick the cleanest 10–20s from the Phase 1 recordings).
