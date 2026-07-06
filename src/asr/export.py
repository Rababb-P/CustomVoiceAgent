"""Merge LoRA weights into the base Whisper model and convert to CTranslate2
so it drops straight into faster-whisper for inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr_finetune.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    adapter_dir = Path(cfg["training"]["output_dir"]) / "final"
    merged_dir = Path(cfg["export"]["merged_dir"])
    ct2_dir = Path(cfg["export"]["ct2_dir"])

    base = WhisperForConditionalGeneration.from_pretrained(cfg["model"]["base"])
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    WhisperProcessor.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))
    print(f"Merged model -> {merged_dir}")

    # Module form of ct2-transformers-converter: findable without venv activation.
    subprocess.run(
        [
            sys.executable, "-m", "ctranslate2.converters.transformers",
            "--model", str(merged_dir),
            "--output_dir", str(ct2_dir),
            "--quantization", cfg["export"]["quantization"],
            "--force",
        ],
        check=True,
    )
    print(f"CTranslate2 model -> {ct2_dir}. Use via src.asr.transcribe.")


if __name__ == "__main__":
    main()
