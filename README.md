# Voice Persona Agent v2 — "AI Rababb"

A voice-to-voice AI that answers questions about me, as me — in my cloned voice,
grounded in my actual resume and project history, and unable to make things up.

Speak into a mic; a Whisper model **fine-tuned on my own speech** transcribes it,
a **LangGraph agent** retrieves facts from a personal corpus via RAG, layered
**guardrails** block prompt injection and hallucinated claims, and a **locally
voice-cloned TTS** speaks the answer back. Total API cost: **$0** — everything
runs locally except the LLM, which rides the Gemini free tier behind a
rate-limited, disk-cached client I wrote to make that survivable.

```
 mic audio ──► VAD ──► fine-tuned Whisper ──► input guard ──► LangGraph agent ◄──► RAG tools
 (browser)  (silero)   (LoRA + CTranslate2)   (heuristics +        │                (Chroma +
                                               flash-lite)         ▼                 bge-small)
 speaker ◄── voice-cloned TTS ◄── sentence chunker ◄── output guard (PII regex +
            (Chatterbox, local)   (streams while LLM     groundedness judge)
                                   still generating)
```

## Why this project is interesting (the 60-second tour)

**1. Custom ASR, not an API call.** I fine-tune `whisper-small` with LoRA
(PEFT, r=32 on attention projections) on 1–2 hours of my own recordings, mixed
with Common Voice so it doesn't forget general English. The win is measurable:
WER on a held-out set of my speech, with a per-term report on vocabulary base
Whisper mangles ("WATonomous", "Reparo", "YOLOv11"). The trained adapter is
merged and exported to CTranslate2 so it runs through faster-whisper at
real-time speeds. ([src/asr/](src/asr/))

**2. The agent can't hallucinate my life.** It's not a stuffed prompt — it's a
LangGraph state machine where retrieval is a tool call and *guards are graph
nodes*. Every factual claim in an answer must be supported by chunks retrieved
that turn; a flash-lite judge checks this, gives the model one regeneration
attempt with the violation as feedback, then falls back to an honest "not
sure". A PII denylist (regex + judge) hard-blocks addresses, IDs, and anything
about third parties, no matter what's asked. ([src/agent/graph.py](src/agent/graph.py),
[src/guardrails/](src/guardrails/))

**3. Every change is gated by evals.** `make eval` scores four suites — ASR
WER, retrieval recall@6/MRR, LLM-as-judge answer quality (judge sees gold
*facts*, never gold answers), and a 30-case red-team suite — writes a
timestamped report, and diffs it against the last run. `make eval-ci` exits
non-zero if WER rises >5% relative, recall drops below 0.85, judge scores drop
>0.3, or injection/PII pass rate dips below 100%. ([evals/](evals/))

**4. Free-tier quota as an engineering constraint.** Gemini's free tier is
~10 requests/min and ~250/day. Every LLM call goes through one wrapper
([src/llm.py](src/llm.py)): client-side sliding-window RPM limiter, exponential
backoff, and an on-disk cache keyed by (model, prompt hash) — so eval reruns
cost zero quota. Cheap classification (guards, judges) runs on `flash-lite`,
which has higher limits; only the agent itself uses `flash`.

**5. Latency is a feature.** The token stream is cut at sentence boundaries
([src/tts/chunker.py](src/tts/chunker.py)) and each sentence is synthesized
while the LLM is still generating, so first audio plays early. Per-stage
timings (VAD, ASR, agent, first-audio) are logged every turn, and
`make bench-tts` benchmarks Chatterbox (voice-cloned) vs Kokoro (fast fallback)
on the current machine.

## Repo tour

| Path | What it is |
|---|---|
| [src/llm.py](src/llm.py) | Shared Gemini client: RPM limiter, backoff, disk cache |
| [src/asr/](src/asr/) | Whisper LoRA fine-tune: data prep → train → CT2 export → transcribe |
| [src/rag/](src/rag/) | Markdown corpus → header-aware chunks → bge-small embeddings → Chroma |
| [src/agent/](src/agent/) | LangGraph graph, tools (`search_life_info`, `list_topics`, `clarify`), persona |
| [src/guardrails/](src/guardrails/) | Input guard, output guard (groundedness + PII), policy allow/denylists |
| [src/tts/](src/tts/) | Chatterbox voice clone + Kokoro fallback behind one async interface |
| [src/server/](src/server/) | FastAPI WebSocket `/converse` (audio↔audio) + `POST /ask`, silero VAD |
| [evals/](evals/) | Four eval suites + [report.py](evals/report.py) aggregator with the regression gate |
| [tests/](tests/) | 42 unit tests — graph runs on a fake chat model: no API key, no quota |
| [data/README.md](data/README.md) | Corpus format + eval JSONL schemas (audio/corpus never committed) |
| [docs/PLAN.md](docs/PLAN.md) | The full phase-by-phase build plan this repo follows |

## Design decisions worth asking me about

- **Guards as graph nodes, not middleware.** Refusals route straight to END via
  conditional edges — an injected prompt never touches the main model or the
  corpus. The graph is the security boundary, and it's unit-testable with fakes.
- **The judge sees gold facts, never gold answers.** Scoring against a
  reference answer rewards parroting; scoring claims against facts measures
  what I actually care about — faithfulness.
- **Streaming vs. groundedness tradeoff.** The groundedness judge needs the
  full answer, so the streaming voice path runs the fast local PII gate per
  sentence, while the full judge stack runs on `POST /ask` and in evals. Chosen
  deliberately: latency for conversation, strictness where it's measured.
- **LoRA + CTranslate2 instead of full fine-tune.** Adapter training fits a
  consumer GPU; merging + int8 CT2 export means inference is identical in cost
  to stock faster-whisper.
- **Boring wrappers.** Chroma is hidden behind [store.py](src/rag/store.py),
  both TTS engines behind one interface, all LLM calls behind
  [llm.py](src/llm.py) — every vendor choice is swappable.

## Running it

```bash
# 1. Install (Python 3.11+). uv works too: uv sync --extra dev --extra rag
make setup

# 2. Configure
cp .env.example .env        # add your free GOOGLE_API_KEY (aistudio.google.com/apikey)

# 3. Write the corpus (markdown in data/corpus/ — stubs are generated) and index it
make ingest

# 4. Talk to it
python -m src.agent.graph "what did you build at hack canada" -v   # text, with node trace
make serve                                                          # then open http://localhost:8000

# Quality gates
make test        # unit tests (no API key needed)
make eval        # all suites, prints comparison vs last run
make redteam     # just the adversarial suite
```

The ASR and TTS phases need my recordings and a GPU:
`make prepare-asr` → hand-correct the transcript TSV → `make train-asr` →
`make export-asr`. TTS voice cloning just needs a clean 10–20s reference clip in
`data/audio/reference/voice_ref.wav`.

## Status

| Phase | State |
|---|---|
| 0 — Scaffolding, LLM wrapper, tooling | ✅ done |
| 1 — ASR fine-tune pipeline | ✅ code complete — awaiting my recordings + GPU run |
| 2 — RAG over life corpus | ✅ done — corpus stubs need my real content |
| 3 — LangGraph agent | ✅ done |
| 4 — Guardrails | ✅ done — red-team suite committed |
| 5 — Eval harness + regression gate | ✅ done |
| 6 — Voice loop (TTS, VAD, WebSocket server) | ✅ done — barge-in is a flagged stretch goal |

Privacy note: recordings, the personal corpus, and the vector index are
gitignored and never leave my machine. The only cloud dependency is the Gemini
free tier, and nothing private goes into it by policy
([src/guardrails/policy.py](src/guardrails/policy.py)).
