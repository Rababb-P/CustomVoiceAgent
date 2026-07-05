# data/

Everything here except this README and the eval fixtures is **gitignored** —
audio recordings and the personal corpus never leave this machine.

```
data/
├── audio/raw/          # your voice recordings (wav/mp3/m4a), 60–120 min total
├── audio/processed/    # segments, corrected transcripts.tsv, HF dataset
├── audio/reference/    # voice_ref.wav — only needed if using optional Chatterbox cloning
├── corpus/             # markdown persona corpus (stubs committed locally only)
├── evals/              # JSONL eval sets (fixtures below ARE committed)
└── .chroma/            # persisted vector index (make ingest)
```

## Corpus format (`data/corpus/**/*.md`)

Markdown with frontmatter. Heading structure matters — the splitter keeps the
heading path in chunk metadata, and `list_sources()` uses it as a table of contents.

```markdown
---
source: projects/reparo
topic: projects
last_updated: 2026-07-05
---
# Reparo
## What it is
...
```

Never put private info here: the corpus is sent to the Gemini free tier, which
may use prompts for training.

## Eval JSONL schemas (`data/evals/`)

**asr_eval.jsonl** — held-out personal speech (audio stays local; only text committed)
```json
{"audio_path": "data/audio/processed/heldout/x.wav", "transcript": "reference text"}
```

**rag_qa.jsonl** — retrieval gold labels (`gold_sources` are paths relative to `data/corpus/`)
```json
{"question": "...", "gold_sources": ["projects/reparo.md"]}
```

**agent_eval.jsonl** — end-to-end quality; judge sees gold *facts*, never gold answers
```json
{"question": "...", "gold_facts": ["fact the answer must be consistent with"], "rubric_notes": "optional"}
```

**adversarial.jsonl** — red-team cases; `must_not_contain` are case-insensitive regexes
```json
{"prompt": "...", "category": "injection|pii_fishing|off_topic|identity", "must_not_contain": ["regex"]}
```

Committed fixtures contain only synthetic content. Grow `rag_qa.jsonl` to ~40
questions and keep `adversarial.jsonl` current as new attack styles show up.
