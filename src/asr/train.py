"""LoRA fine-tune Whisper on the prepared personal dataset.

Uses HF Seq2SeqTrainer with WER (jiwer) as the eval metric, logging to a local
tensorboard dir under runs/. Designed for a single consumer GPU (whisper-small
+ LoRA r=32 fits comfortably in ~8GB with fp16).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config


@dataclass
class Collator:
    """Pad input features and labels separately; mask label padding with -100."""

    processor: object

    def __call__(self, features):

        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        # Trainer re-adds BOS; strip it if present on every row.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr_finetune.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    import evaluate as hf_evaluate
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    base = cfg["model"]["base"]
    processor = WhisperProcessor.from_pretrained(
        base, language=cfg["model"]["language"], task=cfg["model"]["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(base)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
        ),
    )
    model.print_trainable_parameters()

    ds = load_from_disk(str(Path(cfg["data"]["processed_dir"]) / "hf_dataset"))

    def preprocess(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names, num_proc=1)

    wer_metric = hf_evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    t = cfg["training"]
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=t["output_dir"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            learning_rate=float(t["learning_rate"]),
            warmup_steps=t["warmup_steps"],
            num_train_epochs=t["num_train_epochs"],
            fp16=t["fp16"],
            eval_strategy=t["eval_strategy"],
            # Ship the best epoch, not whichever state training happened to
            # end on — mid-run instability otherwise overwrites a good adapter.
            save_strategy=t["eval_strategy"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=t["predict_with_generate"],
            generation_max_length=t["generation_max_length"],
            logging_steps=t["logging_steps"],
            save_total_limit=t["save_total_limit"],
            seed=t["seed"],
            report_to=["tensorboard"],
            remove_unused_columns=False,
            label_names=["labels"],
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    final_dir = Path(t["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved LoRA adapter to {final_dir}. Next: make export-asr")


if __name__ == "__main__":
    main()
