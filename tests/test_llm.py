import pytest

import src.llm as llm


def _cfg(tmp_path):
    return {
        "llm": {
            "agent_model": "fake-model",
            "guard_model": "fake-model",
            "judge_model": "fake-model",
            "rpm": {"fake-model": 1000},
            "max_retries": 3,
            "cache_dir": str(tmp_path),  # absolute, so ROOT / dir resolves to it
        }
    }


def test_response_cached_on_disk(tmp_path):
    cfg = _cfg(tmp_path)
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return "answer one"

    assert llm.generate("hello", config=cfg, _call=fake) == "answer one"
    assert llm.generate("hello", config=cfg, _call=fake) == "answer one"
    assert len(calls) == 1  # second hit came from cache
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_keyed_by_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    llm.generate("prompt A", config=cfg, _call=lambda p: "A")
    assert llm.generate("prompt B", config=cfg, _call=lambda p: "B") == "B"
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_backoff_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    cfg = _cfg(tmp_path)
    attempts = []

    def flaky(prompt):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("429 rate limit")
        return "finally"

    assert llm.generate("flaky prompt", config=cfg, _call=flaky) == "finally"
    assert len(attempts) == 3


def test_exhausted_retries_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    cfg = _cfg(tmp_path)

    def always_fails(prompt):
        raise RuntimeError("429")

    with pytest.raises(RuntimeError, match="failed after retries"):
        llm.generate("doomed prompt", config=cfg, _call=always_fails)


def test_rpm_limiter_blocks_over_limit(monkeypatch):
    limiter = llm._RpmLimiter(rpm=2)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    # Third acquire within the window must wait — sleep gets called; drain one
    # stamp manually so the loop can exit.
    limiter.acquire()
    limiter.acquire()

    original_sleep_count = len(sleeps)

    def fake_sleep(s):
        sleeps.append(s)
        limiter._stamps.pop(0)

    monkeypatch.setattr(llm.time, "sleep", fake_sleep)
    limiter.acquire()
    assert len(sleeps) > original_sleep_count
