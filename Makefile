# Voice Persona Agent v2. `make help` lists targets.
PY ?= python

help:
	@echo "Targets:"
	@echo "  setup        install core + dev deps (uv sync or pip)"
	@echo "  lint         ruff check + format check"
	@echo "  test         pytest"
	@echo "  synth-asr    generate sentences + render multi-voice training audio"
	@echo "  prepare-asr  assemble HF dataset (synthetic + TechVoice + Common Voice)"
	@echo "  train-asr    LoRA fine-tune Whisper on the assembled dataset"
	@echo "  export-asr   merge LoRA + convert to CTranslate2"
	@echo "  eval-asr     WER base vs fine-tuned + custom-vocab report"
	@echo "  ingest       build the Chroma index from data/corpus"
	@echo "  eval-rag     recall@k / MRR on rag_qa.jsonl"
	@echo "  eval-agent   LLM-as-judge over end-to-end answers"
	@echo "  redteam      adversarial suite (injections, PII fishing)"
	@echo "  eval         all suites -> evals/results/<ts>.json + comparison"
	@echo "  eval-ci      eval + non-zero exit on regression"
	@echo "  bench-tts    Chatterbox vs Kokoro synthesis speed"
	@echo "  serve        FastAPI voice server on :8000"

setup:
	uv sync --extra dev --extra rag || $(PY) -m pip install -e ".[dev,rag]"

lint:
	ruff check src evals tests && ruff format --check src evals tests

test:
	$(PY) -m pytest -q

# ---- Phase 1: ASR ----
synth-asr:
	$(PY) -m src.asr.gen_sentences --config configs/asr_finetune.yaml
	$(PY) -m src.asr.synthesize --config configs/asr_finetune.yaml

prepare-asr:
	$(PY) -m src.asr.prepare_data --synthetic --config configs/asr_finetune.yaml

train-asr:
	$(PY) -m src.asr.train --config configs/asr_finetune.yaml

export-asr:
	$(PY) -m src.asr.export --config configs/asr_finetune.yaml

eval-asr:
	$(PY) -m evals.run_asr_eval

# ---- Phase 2: RAG ----
ingest:
	$(PY) -m src.rag.ingest --config configs/rag.yaml

eval-rag:
	$(PY) -m evals.run_rag_eval

# ---- Phases 3-5 ----
eval-agent:
	$(PY) -m evals.run_agent_eval

redteam:
	$(PY) -m evals.run_redteam

eval:
	$(PY) -m evals.report

eval-ci:
	$(PY) -m evals.report --ci

# ---- Phase 6 ----
bench-tts:
	$(PY) -m src.tts.speak --bench

serve:
	$(PY) -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000

.PHONY: help setup lint test synth-asr prepare-asr train-asr export-asr eval-asr ingest \
        eval-rag eval-agent redteam eval eval-ci bench-tts serve
