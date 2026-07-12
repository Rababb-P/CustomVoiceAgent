# Reproducing the CPU recovery run

This run rebuilds a Whisper-small LoRA adapter because the old notebook's
model files were not available. Its measurements belong to this new run;
the notebook's historical numbers do not describe these weights.

## Data and configuration

[asr_retrain_cpu.yaml](../configs/asr_retrain_cpu.yaml) keeps Whisper-small,
rank 32, alpha 64, and adapters on the attention query/value projections.
The CPU run uses batches of two, four-batch gradient accumulation, three epochs,
and a peak learning rate of 0.0002 with 20 warmup updates.

The 160 [source sentences](asr_retrain_sentences.txt) were authored by the
assistant for this run. They contain interview questions, technical explanations,
and general speech; they are synthetic training text, not verified biography.
They cover WATonomous, Reparo, YOLOv11, ROS2, Slurm, Thevenin, Waterloo, and LangGraph.

Sentence IDs determine a reproducible split. Kokoro renders 128 training
sentences in two voices each and 32 validation sentences in separate held-out
voices. Speed changes and noise apply only to training audio. The training set
also contains the first 256 examples of the Common Voice 17 English training
split, yielding **512 training examples and 32 synthetic validation examples**.
The first 32 examples of the Common Voice test split provide a small real-speech
check. These test examples do not participate in optimization or epoch selection.

This is a small recovery experiment, not a broad speech-recognition benchmark.
Synthetic validation selects the best epoch, so its score is not an independent
test estimate. Per-term sample counts are small. Real-world microphone audio,
accents, and noise may behave differently.

Sources:
[Whisper-small](https://huggingface.co/openai/whisper-small),
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), and the
[Common Voice mirror](https://huggingface.co/datasets/fixie-ai/common_voice_17_0).
The saved environment report records the exact source revisions and installed
package versions. Audio and intermediate datasets stay gitignored.

## Commands

From the repository root, install the ASR extra and Kokoro into your virtual
environment. Consult the published adapter's environment report for the versions
used in this run. No LLM API key is needed to regenerate the supplied sentences.

```bash
python -m pip install -e ".[asr]" kokoro
python -m scripts.prepare_asr_retrain
python -m scripts.evaluate_asr_retrain --base-only
python -m src.asr.train --config configs/asr_retrain_cpu.yaml
python -m src.asr.export --config configs/asr_retrain_cpu.yaml
python -m scripts.evaluate_asr_retrain
```

For this run, PyTorch used eight CPU threads during training and four during
Kokoro synthesis. The training script logs progress and saves the best adapter
to `runs/asr-retrain-cpu/final/`, with epoch metrics in `training_metrics.json`.
An adapter contains the learned updates; loading it also requires the original
Whisper-small base model. Export merges the updates and produces an int8
CTranslate2 model at the path used by the voice server.

Evaluation compares base, base with hotword hints, and the fine-tuned model.
All use int8 CTranslate2, English, greedy decoding, and no previous-text
conditioning. The report includes raw WER, normalized WER (lowercase, punctuation
removed, whitespace collapsed), CER, literal term accuracy, and predictions.
The fine-tuned comparison does not use hotword hints. Training's per-epoch WER
uses Hugging Face generation in float32, so it may differ from the exported
int8 model's score.

The evaluation command also updates `data/evals/asr_eval.jsonl` to point
`make eval-asr` at this run's locally generated audio. That existing suite uses
its own decoding defaults; use the saved report for the matched comparison above.

## Using a published adapter

Export accepts an explicit adapter folder, so downloaded weights do not need
to be placed inside a training run directory:

```bash
python -m src.asr.export --config configs/asr_retrain_cpu.yaml \
  --adapter-dir artifacts/asr/whisper-small-lora
```

Then start the voice server normally. Its configured model directory is
`models/whisper-personal-ct2`.
