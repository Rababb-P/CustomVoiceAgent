"""Synthetic ASR data pipeline: sentence validation, voice-split discipline,
augmentation, hotword wiring. No network, no TTS model, no GPU."""

import numpy as np

from src.asr.gen_sentences import find_terms, generate_llm_sentences, validate
from src.asr.synthesize import add_noise, assign_voices, plan_renders, split_sentences

VOCAB = ["WATonomous", "Reparo", "YOLOv11", "ROS2"]

SYN_CFG = {
    "train_voices": ["af_heart", "am_michael", "bf_emma"],
    "heldout_voices": ["af_sky", "am_adam"],
    "heldout_sentence_fraction": 0.25,
    "speeds": [0.9, 1.0, 1.1],
    "noise_fraction": 0.5,
    "noise_snr_db": [15, 30],
}


def _rows(n=8):
    return [
        {
            "id": f"{i:010x}",
            "text": f"sentence number {i} about Reparo and robots today",
            "terms": ["Reparo"],
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------- gen_sentences


def test_validate_requires_term_when_asked():
    plain = "I really like building robots with my friends."
    assert validate("I led the perception team at WATonomous last fall.", VOCAB, require_term=True)
    assert validate(plain, VOCAB, require_term=True) is None
    assert validate(plain, VOCAB, require_term=False) == []


def test_validate_rejects_bad_lengths_and_markdown():
    markdown = "Here is **Reparo** in markdown formatting style today"
    assert validate("too short", VOCAB, require_term=False) is None
    assert validate("word " * 40, VOCAB, require_term=False) is None
    assert validate(markdown, VOCAB, require_term=True) is None


def test_find_terms_case_insensitive():
    assert find_terms("we used yolov11 and ros2 on the car", VOCAB) == ["YOLOv11", "ROS2"]


def test_generate_llm_sentences_dedupes_and_hits_target():
    cfg = {"data": {"synthetic": {"n_llm_sentences": 4, "sentences_per_call": 3},
                    "custom_vocab": VOCAB}}
    batches = iter([
        "I built Reparo at a hackathon last winter with two teammates.\n"
        "I built Reparo at a hackathon last winter with two teammates.\n"  # duplicate
        "bad line",
        "We ship YOLOv11 models to the WATonomous perception stack every week.\n"
        "My ROS2 nodes handle sensor fusion for the autonomy pipeline.\n"
        "Reparo ended up winning the sustainability category at the event.",
    ])

    def fake_generate(prompt, role):
        return next(batches)

    rows = generate_llm_sentences(cfg, generate_fn=fake_generate)
    assert len(rows) == 4
    texts = [r["text"] for r in rows]
    assert len(set(texts)) == 4  # deduped
    assert all(r["terms"] for r in rows)


# ------------------------------------------------------------------ synthesize


def test_split_is_deterministic_and_sized():
    rows = _rows(8)
    s1 = split_sentences(rows, 0.25)
    s2 = split_sentences(list(reversed(rows)), 0.25)
    assert s1 == s2
    assert sum(1 for v in s1.values() if v == "val") == 2


def test_val_renders_use_only_heldout_voices():
    renders = plan_renders(_rows(8), SYN_CFG)
    train_voices = {r["voice"] for r in renders if r["split"] == "train"}
    val_voices = {r["voice"] for r in renders if r["split"] == "val"}
    assert train_voices <= set(SYN_CFG["train_voices"])
    assert val_voices <= set(SYN_CFG["heldout_voices"])
    assert train_voices.isdisjoint(val_voices)


def test_val_renders_never_augmented():
    renders = plan_renders(_rows(12), SYN_CFG)
    for r in renders:
        if r["split"] == "val":
            assert r["speed"] == 1.0 and not r["noisy"]


def test_assign_voices_deterministic():
    voices = ["a", "b", "c", "d"]
    assert assign_voices("00000000ff", voices) == assign_voices("00000000ff", voices)
    assert len(assign_voices("00000000ff", voices)) == 2


def test_add_noise_hits_target_snr():
    rng = np.random.default_rng(0)
    signal = np.sin(np.linspace(0, 400 * np.pi, 16000)).astype(np.float32)
    noisy = add_noise(signal, snr_db=20, rng=rng)
    assert noisy.shape == signal.shape
    noise = noisy - signal
    snr = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
    assert 18 < snr < 22


# -------------------------------------------------------------------- hotwords


def test_hotwords_built_from_config_vocab():
    from src.asr.transcribe import _hotwords

    _hotwords.cache_clear()
    hw = _hotwords()
    assert hw is not None
    assert "WATonomous" in hw and "Reparo" in hw
