"""Training checks on tiny local models; no downloads or ASR training data."""

from types import SimpleNamespace

import pytest

from src.asr.train import Collator, lr_lambda_linear, train_one_epoch

torch = pytest.importorskip("torch")


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_features, labels):
        return SimpleNamespace(loss=((self.weight * input_features - labels) ** 2).mean())


def test_partial_accumulation_matches_explicit_grouped_batches():
    batches = [
        {"input_features": torch.tensor([1.0]), "labels": torch.tensor([target])}
        for target in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]
    actual, expected = ScalarModel(), ScalarModel()
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    train_one_epoch(
        actual,
        batches,
        optimizer,
        "cpu",
        accumulation_steps=2,
        scheduler=scheduler,
        max_grad_norm=float("inf"),
    )
    for start in range(0, len(batches), 2):
        group = batches[start : start + 2]
        reference_optimizer.zero_grad()
        loss = expected(
            torch.cat([b["input_features"] for b in group]),
            torch.cat([b["labels"] for b in group]),
        ).loss
        loss.backward()
        reference_optimizer.step()

    torch.testing.assert_close(actual.weight, expected.weight)
    assert scheduler.last_epoch == 3


def test_amp_overflow_does_not_advance_weights_or_schedule():
    model = ScalarModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu")
    batches = [{"input_features": torch.tensor([float("inf")]), "labels": torch.ones(1)}]
    before = model.weight.detach().clone()
    scale = scaler.get_scale()

    train_one_epoch(model, batches, optimizer, "cpu", scaler=scaler, scheduler=scheduler)

    torch.testing.assert_close(model.weight, before)
    assert scheduler.last_epoch == 0
    assert scaler.get_scale() < scale


def test_collator_strips_decoder_start_and_masks_padding():
    class Padder:
        def pad(self, features, return_tensors):
            if "input_features" in features[0]:
                return {"input_features": torch.stack([f["input_features"] for f in features])}
            sequences = [torch.tensor(f["input_ids"]) for f in features]
            ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
            return {"input_ids": ids, "attention_mask": ids.ne(0).long()}

        def convert_tokens_to_ids(self, token):
            assert token == "<|startoftranscript|>"
            return 1

    processor = SimpleNamespace(feature_extractor=Padder(), tokenizer=Padder())
    features = [
        {"input_features": torch.ones(2, 4), "labels": [1, 3, 4]},
        {"input_features": torch.ones(2, 4), "labels": [1, 5]},
    ]
    batch = Collator(processor)(features)
    assert batch["labels"].tolist() == [[3, 4], [5, -100]]
    features[0]["labels"] = [3, 4]
    features[1]["labels"] = [5]
    assert Collator(processor, decoder_start_token_id=1)(features)["labels"].tolist() == [
        [3, 4],
        [5, -100],
    ]


def test_whisper_lora_updates_only_adapters_and_round_trips(tmp_path):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = transformers.WhisperConfig(
        vocab_size=16,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_source_positions=4,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    torch.manual_seed(42)
    base = transformers.WhisperForConditionalGeneration(config)
    base_state = {name: p.clone() for name, p in base.state_dict().items()}
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    assert all(p.requires_grad == ("lora_" in name) for name, p in model.named_parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    batch = {"input_features": torch.randn(2, 4, 8), "labels": torch.tensor([[3, 2], [4, 2]])}
    train_one_epoch(model, [batch, batch], optimizer, "cpu")

    changed = [name for name, p in model.named_parameters() if not torch.equal(p, before[name])]
    assert changed and all("lora_" in name for name in changed)
    model.eval()
    expected = model(**batch).logits.detach()
    model.save_pretrained(tmp_path)
    restored_base = transformers.WhisperForConditionalGeneration(config)
    restored_base.load_state_dict(base_state)
    restored = peft.PeftModel.from_pretrained(restored_base, tmp_path)
    restored.eval()
    torch.testing.assert_close(restored(**batch).logits, expected)
    merged = restored.merge_and_unload()
    torch.testing.assert_close(merged(**batch).logits, expected)


def test_learning_rate_warmup_and_decay():
    schedule = lr_lambda_linear(warmup_steps=2, total_steps=6)
    assert [schedule(step) for step in (0, 1, 2, 4, 6, 7)] == [0, 0.5, 1, 0.5, 0, 0]
