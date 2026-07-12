"""LoRA fine-tune Whisper on the prepared personal dataset — raw PyTorch loop.

No Trainer: the loop below owns the optimizer, LR schedule, AMP scaling,
gradient accumulation, evaluation, and best-epoch selection directly. WER
(jiwer) is the eval metric, logged to a local tensorboard dir under runs/.
Designed for a single consumer GPU (whisper-small + LoRA r=32 fits comfortably
in ~8GB with fp16).

The same loop, narrated cell-by-cell, lives in notebooks/asr_finetune_pytorch.ipynb.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config

logger = logging.getLogger(__name__)


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
        # The model prepends BOS itself; strip it if present on every row.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def lr_lambda_linear(warmup_steps: int, total_steps: int):
    """Linear warmup to peak LR, then linear decay to zero."""

    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return f


def evaluate_wer(model, loader, processor, device, max_length: int) -> float:
    """Greedy-decode the val split and score word error rate."""
    import jiwer
    import torch

    model.eval()
    preds: list[str] = []
    refs: list[str] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["input_features"].to(device, dtype=model.dtype)
            generated = model.generate(input_features=features, max_length=max_length)
            preds.extend(processor.batch_decode(generated, skip_special_tokens=True))
            labels = batch["labels"].masked_fill(
                batch["labels"] == -100, processor.tokenizer.pad_token_id
            )
            refs.extend(processor.batch_decode(labels, skip_special_tokens=True))
    model.train()
    return jiwer.wer(refs, preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr_finetune.yaml")
    parser.add_argument("--limit", type=int, help="train/val on the first N examples (smoke)")
    parser.add_argument("--output-dir", help="override training.output_dir (smoke)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    t = cfg["training"]
    out_dir = Path(args.output_dir or t["output_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = bool(t.get("fp16", True)) and device == "cuda"
    torch.manual_seed(t.get("seed", 42))
    logger.info("training on %s (fp16=%s)", device, use_fp16)

    # ---- model + LoRA -------------------------------------------------------
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
    model.to(device)

    # ---- data ---------------------------------------------------------------
    ds = load_from_disk(str(Path(cfg["data"]["processed_dir"]) / "hf_dataset"))
    if args.limit:
        ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(args.limit, len(ds["validation"]))))

    def preprocess(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names, num_proc=1)
    ds.set_format("torch", columns=["input_features", "labels"], output_all_columns=True)

    collate = Collator(processor)
    batch_size = t["per_device_train_batch_size"]
    train_loader = DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(ds["validation"], batch_size=batch_size, collate_fn=collate)

    # ---- optimizer, schedule, AMP -------------------------------------------
    accum = t["gradient_accumulation_steps"]
    epochs = t["num_train_epochs"]
    steps_per_epoch = -(-len(train_loader) // accum)  # ceil
    total_steps = steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=float(t["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_linear(t["warmup_steps"], total_steps)
    )
    scaler = torch.amp.GradScaler(device, enabled=use_fp16)
    writer = SummaryWriter(str(out_dir))

    # ---- the loop ------------------------------------------------------------
    final_dir = out_dir / "final"
    best_wer = float("inf")
    step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device, enabled=use_fp16):
                loss = model(input_features=features, labels=labels).loss / accum
            scaler.scale(loss).backward()

            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                if step % t.get("logging_steps", 25) == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        "step %d/%d  loss %.4f  lr %.2e", step, total_steps,
                        loss.item() * accum, lr_now,
                    )
                    writer.add_scalar("train/loss", loss.item() * accum, step)
                    writer.add_scalar("train/lr", lr_now, step)

        wer = evaluate_wer(model, val_loader, processor, device, t["generation_max_length"])
        writer.add_scalar("eval/wer", wer, epoch)
        logger.info("epoch %d/%d  val WER %.4f", epoch, epochs, wer)
        if wer < best_wer:
            # Best-epoch selection: only the best adapter ever reaches final/
            # (a mid-run blowup must not overwrite a good earlier epoch).
            best_wer = wer
            model.save_pretrained(str(final_dir))
            processor.save_pretrained(str(final_dir))
            logger.info("new best (WER %.4f) -> %s", wer, final_dir)

    writer.close()
    print(f"Saved best LoRA adapter (WER {best_wer:.4f}) to {final_dir}. Next: make export-asr")


if __name__ == "__main__":
    main()
