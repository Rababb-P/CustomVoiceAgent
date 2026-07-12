"""LoRA fine-tune Whisper on the prepared personal dataset — raw PyTorch loop.

No Trainer: the loop below owns the optimizer, LR schedule, AMP scaling,
gradient accumulation, evaluation, and best-epoch selection directly. WER
(jiwer) is the eval metric, logged to a local tensorboard dir under runs/.
Designed for a single consumer GPU (whisper-small + LoRA r=32 fits comfortably
in ~8GB with fp16).

Start with train_one_epoch below; docs/ASR_TRAINING.md maps it to the math.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config

logger = logging.getLogger(__name__)


def load_processor(base: str, language: str, task: str):
    """Configure both tokenizer attributes and its fast-tokenizer prefix template."""
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(base, language=language, task=task)
    # Some versions set the attributes without rebuilding the fast backend's
    # template. An explicit call ensures labels contain language + task tokens.
    processor.tokenizer.set_prefix_tokens(language=language, task=task, predict_timestamps=False)
    return processor


@dataclass
class Collator:
    """Pad input features and labels separately; mask label padding with -100."""

    processor: object
    decoder_start_token_id: int | None = None

    def __call__(self, features):

        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        # Whisper shifts labels right and prepends decoder_start_token_id itself.
        # Its tokenizer.bos_token_id is a different token (<|endoftext|>).
        start_id = self.decoder_start_token_id
        if start_id is None:
            start_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == start_id).all().item():
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    accumulation_steps=1,
    scaler=None,
    scheduler=None,
    max_grad_norm=1.0,
) -> float:
    """Forward -> loss -> backward -> update; return mean batch loss.

    With accumulation_steps=1 and scaler=None, this is the basic PyTorch loop.
    Accumulation averages several batch losses before one parameter update.
    An optional GradScaler protects fp16 gradients from numerical underflow.
    """
    import torch

    if accumulation_steps < 1 or len(loader) == 0:
        raise ValueError("Need a nonempty loader and accumulation_steps >= 1")

    model.train()  # Enables training behavior such as dropout; does not unfreeze weights.
    use_amp = scaler is not None and scaler.is_enabled()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        if i % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)  # Clear the previous update's gradients.
            # The final group may contain fewer batches than accumulation_steps.
            group_size = min(accumulation_steps, len(loader) - i)

        features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
            outputs = model(input_features=features, labels=labels)  # Forward pass.
            loss = outputs.loss  # Token cross-entropy, computed by Whisper.
        total_loss += loss.detach().item()
        if (i + 1) % 25 == 0 or i + 1 == len(loader):
            logger.info("batch %d/%d  mean loss %.4f", i + 1, len(loader), total_loss / (i + 1))
        loss = loss / group_size

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()  # Autograd accumulates gradients for trainable parameters.

        if (i + 1) % accumulation_steps != 0 and i + 1 != len(loader):
            continue  # Keep accumulating before updating the parameters.

        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_grad_norm
        )
        if use_amp:
            old_scale = scaler.get_scale()
            scaler.step(optimizer)  # Calls optimizer.step() unless gradients overflow.
            scaler.update()
            did_update = scaler.get_scale() >= old_scale
        else:
            optimizer.step()  # AdamW updates only the parameters given to it.
            did_update = True
        if scheduler is not None and did_update:
            scheduler.step()  # Advance LR only when the parameters were updated.

    return total_loss / len(loader)


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
    dataset_dir = Path(cfg["data"]["processed_dir"]) / "hf_dataset"
    if not dataset_dir.is_dir():
        parser.error(f"Prepared dataset missing: {dataset_dir}. Run make prepare-asr first.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from transformers import WhisperForConditionalGeneration

    t = cfg["training"]
    out_dir = Path(args.output_dir or t["output_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = bool(t.get("fp16", True)) and device == "cuda"
    torch.manual_seed(t.get("seed", 42))
    logger.info("training on %s (fp16=%s)", device, use_fp16)

    # ---- model + LoRA -------------------------------------------------------
    base = cfg["model"]["base"]
    processor = load_processor(base, language=cfg["model"]["language"], task=cfg["model"]["task"])
    model = WhisperForConditionalGeneration.from_pretrained(base)
    decoder_start_token_id = model.config.decoder_start_token_id
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.generation_config.forced_decoder_ids = None

    lora = cfg["lora"]
    # PEFT freezes the base weights W and adds trainable A and B to q_proj/v_proj.
    # Each adapted linear layer computes W x + (alpha / r) B A x (plus dropout).
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
    ds = load_from_disk(str(dataset_dir))
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
    prefix = processor.tokenizer.prefix_tokens
    if ds["train"][0]["labels"][: len(prefix)] != prefix:
        raise ValueError("Training labels are missing the configured language/task prefix")
    ds.set_format("torch", columns=["input_features", "labels"], output_all_columns=True)

    collate = Collator(processor, decoder_start_token_id)
    batch_size = t["per_device_train_batch_size"]
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ds["validation"], batch_size=batch_size, collate_fn=collate)

    # ---- optimizer, schedule, AMP -------------------------------------------
    accum = t["gradient_accumulation_steps"]
    epochs = t["num_train_epochs"]
    steps_per_epoch = -(-len(train_loader) // accum)  # ceil
    total_steps = steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(t["learning_rate"]),
        weight_decay=float(t.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_linear(t["warmup_steps"], total_steps)
    )
    scaler = torch.amp.GradScaler(device) if use_fp16 else None
    writer = SummaryWriter(str(out_dir))

    # ---- the loop ------------------------------------------------------------
    final_dir = out_dir / "final"
    best_wer = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            accumulation_steps=accum,
            scaler=scaler,
            scheduler=scheduler,
        )
        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        wer = evaluate_wer(model, val_loader, processor, device, t["generation_max_length"])
        writer.add_scalar("eval/wer", wer, epoch)
        logger.info("epoch %d/%d  loss %.4f  val WER %.4f", epoch, epochs, loss, wer)
        if wer < best_wer:
            # Best-epoch selection: only the best adapter ever reaches final/
            # (a mid-run blowup must not overwrite a good earlier epoch).
            best_wer = wer
            model.save_pretrained(str(final_dir))
            processor.save_pretrained(str(final_dir))
            logger.info("new best (WER %.4f) -> %s", wer, final_dir)
        history.append({"epoch": epoch, "train_loss": loss, "validation_wer": wer})
        (out_dir / "training_metrics.json").write_text(
            json.dumps(
                {
                    "base_model": base,
                    "device": device,
                    "seed": t.get("seed", 42),
                    "train_examples": len(ds["train"]),
                    "validation_examples": len(ds["validation"]),
                    "best_validation_wer": best_wer,
                    "epochs": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    writer.close()
    print(f"Saved best LoRA adapter (WER {best_wer:.4f}) to {final_dir}. Next: make export-asr")


if __name__ == "__main__":
    main()
