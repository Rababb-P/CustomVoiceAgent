---
base_model: openai/whisper-small
library_name: peft
pipeline_tag: automatic-speech-recognition
language: en
---

# Whisper-small domain LoRA adapter

An actual CPU fine-tune of Whisper-small, trained on September 17, 2026.
This is a new recovery run; it is not the model measured in the historical notebook.

The adapter learns domain vocabulary including WATonomous, Reparo, YOLOv11,
ROS2, Slurm, Thevenin, Waterloo, and LangGraph. Base weights remain frozen.
Rank is 32, alpha is 64, and adapters target query/value projections in both
the encoder and decoder. There are 3,538,944 trainable adapter parameters.

## Data and selection

512 training examples: 256 Kokoro renders of assistant-authored text plus
256 Common Voice 17 English training clips. Validation contains 32 held-out
sentences spoken by held-out Kokoro voices; a separate 32-clip Common Voice
test slice checks real speech. There are no exact transcript overlaps between
the training set and the real test slice.

Training ran for 3 epochs on a Ryzen 9 5900X CPU. The published
adapter is epoch 2, selected by the lowest Hugging Face validation WER
(1.02%). See [training_metrics.json](training_metrics.json).

## Matched exported-model evaluation

| Model | Synthetic validation normalized WER | Real test normalized WER |
|---|---:|---:|
| Whisper-small | 5.78% | 21.89% |
| Whisper-small + hotwords | 2.04% | 21.13% |
| Fine-tuned Whisper-small | 1.02% | 21.51% |

All variants use int8 CTranslate2, English, beam size 1, temperature 0, and
no previous-text conditioning. The fine-tuned model has no hotword hints.
Normalized WER lowercases text, removes punctuation, and collapses whitespace.
[evaluation.json](evaluation.json) includes raw WER, CER, term accuracy, and predictions.

These are small evaluation sets. Synthetic validation selected the checkpoint,
so it is not an independent test estimate. The results do not establish broad
accuracy across speakers, accents, microphones, or noise conditions.

## Load or export

This folder contains adapter weights and the processor, not a copy of the base
model. Loading it requires `openai/whisper-small`:

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

adapter_dir = "artifacts/asr/whisper-small-lora"
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, adapter_dir).eval()
processor = WhisperProcessor.from_pretrained(adapter_dir)
```

From the repository root:

```bash
python -m src.asr.export --config configs/asr_retrain_cpu.yaml --adapter-dir artifacts/asr/whisper-small-lora
```

The [GitHub release](https://github.com/Rababb-P/CustomVoiceAgent/releases/tag/asr-retrain-2026-09-17)
also provides a merged int8 CTranslate2 ZIP that needs no base-model download.
Extract its `whisper-personal-ct2` folder into `models/` and start the voice server.

The [reproduction guide](../../../docs/ASR_RETRAINING.md) links the source sentences
and commands. [environment.json](environment.json) records package versions and
upstream model/data revisions; [training_config.yaml](training_config.yaml) records
hyperparameters. Source audio stays local and is not included in this package.

The original Whisper license notice is included as [WHISPER_LICENSE](WHISPER_LICENSE).
