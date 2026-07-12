# Explaining the Whisper training loop

The runnable loop is `train_one_epoch` in [train.py](../src/asr/train.py).
It uses PyTorch directly; PEFT installs and saves the LoRA layers.

## The core steps

With one batch per update and mixed precision disabled, the loop is:

```python
model.train()
for batch in dataloader:
    audio = batch["input_features"].to(device)
    labels = batch["labels"].to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_features=audio, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

- `input_features` contains log-mel spectrograms made from 16 kHz audio.
  Preprocessing applies an STFT, mel filters, and a log transform.
- `labels` contains transcript token IDs. Whisper shifts these right to form
  decoder inputs, so each position learns to predict the next token.
- The loss is token cross-entropy: it penalizes low probability on the correct
  token. Padding is replaced with `-100` so it contributes no loss. The collator
  removes the initial decoder-start token because Whisper adds it when shifting.
- `backward()` uses the computation graph to accumulate gradients. Frozen
  weights get no parameter gradients, but gradients can still flow through
  their operations to reach trainable adapters earlier in the network.
- `step()` updates the parameters given to AdamW. The rule
  `w_new = w_old - learning_rate * gradient` describes ordinary gradient descent;
  AdamW also uses moving averages of gradients and squared gradients, with
  separate weight decay.

`model.train()` enables behavior such as dropout. It does not change
`requires_grad` or update parameters by itself.

## What LoRA changes

A linear layer has `W` of shape `(out_dim, in_dim)`. LoRA freezes `W` and learns:

```text
A: (rank, in_dim)
B: (out_dim, rank)
delta_W = (alpha / rank) * B @ A
y = W @ x + (alpha / rank) * B @ A @ x
```

For batched row vectors, the same forward calculation is:

```python
base_out = base_layer(x)
lora_out = x @ A.T @ B.T
return base_out + (alpha / rank) * lora_out
```

The actual PEFT layer also applies configured dropout to the adapter input
during training. `B` starts at zero, so the initial adapter contributes zero.
Only the adapter parameters have `requires_grad=True`, and AdamW receives
only those parameters. We keep PEFT so the saved adapter still works with
[export.py](../src/asr/export.py), which merges it into Whisper for inference.

For a 4096 by 4096 layer and rank 8, full tuning updates 16,777,216 weights;
LoRA updates 65,536. This project's existing setting is **rank 32**, alpha 64,
on attention `q_proj` and `v_proj` layers. Rank 8 is an example, not a requirement;
Whisper-small uses model dimension 768. LoRA limits the form of the update,
but does not guarantee that general speech accuracy will be preserved.

## The extra code in the real loop

- **Gradient accumulation:** two batches share one update in the default
  config. Gradients are cleared at the start of each group and losses are
  divided by its actual batch count, including a short final group.
- **Mixed precision:** on CUDA, autocast and GradScaler reduce memory use and
  protect gradients from underflow. CPU uses ordinary `loss.backward()` and
  `optimizer.step()`. An overflow skips both the weight update and LR step.
- **Gradient clipping:** limits the gradient norm before an update.
- **Learning-rate schedule:** warms up, then decays linearly.
- **Validation:** generated transcripts are scored with word error rate after
  each epoch. Only an improved validation WER replaces `runs/asr/final`.
  TensorBoard receives mean batch loss, learning rate, and WER per epoch.

To study the simplest execution path, set `gradient_accumulation_steps: 1`
and `fp16: false` in a copy of the YAML config. That uses more GPU memory for
the same batch size. It does not require retraining an existing adapter.

## Attention terminology

Whisper's **encoder** uses audio self-attention. Its **decoder** uses causal
text self-attention followed by cross-attention to the encoded audio.
In `softmax(Q @ K.T / sqrt(d_k)) @ V`, queries and keys determine attention
weights, and values are combined using those weights. Causal masking prevents
decoder positions from seeing future transcript tokens during training.

## Saved model status

The executed [notebook](../notebooks/asr_finetune_pytorch.ipynb) records a
previous GPU run and reports results for a full fine-tune. Those recorded
results are not a new evaluation of this code. At this revision, the adapter,
exported model, and prepared training dataset are absent from this checkout;
`runs/` and `models/` are gitignored. Notebook output alone cannot restore weights.

Restore the existing adapter plus processor files into `runs/asr/final/`, then
run `python -m src.asr.export`. If no copy exists, prepare paired audio/transcripts
and run `python -m src.asr.train`. A random test model is not a trained ASR artifact.

References: [PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora),
[PyTorch mixed precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
