"""Shared Gemini client wrapper. Every LLM call in the project goes through here.

Free-tier quotas are tight (~10 RPM / ~250 RPD on gemini-2.5-flash), so this module
enforces three things globally:

1. Client-side RPM limiting  — one shared token-bucket limiter per model.
2. Exponential backoff       — retries on 429/5xx are handled by the chat model's
                               max_retries plus the limiter keeping us under quota.
3. On-disk response cache    — keyed by (model, prompt hash). Cache hits cost zero
                               quota, which is what makes repeated eval runs free.

Two entry points:
- get_chat_model(role)   -> a ChatGoogleGenerativeAI for LangChain/LangGraph use
                            (agent, tools binding, streaming). Rate-limited + cached
                            via LangChain's global SQLite LLM cache.
- generate(prompt, role) -> plain text completion for guards and judges, with our
                            own JSON file cache (simpler to inspect than SQLite).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from src.config import ROOT, load_config

load_dotenv()
logger = logging.getLogger(__name__)

_lock = threading.Lock()
_limiters: dict[str, _RpmLimiter] = {}
_chat_models: dict[tuple, object] = {}
_sqlite_cache_set = False


class _RpmLimiter:
    """Blocking requests-per-minute limiter (sliding window)."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._stamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._stamps = [t for t in self._stamps if now - t < 60]
                if len(self._stamps) < self.rpm:
                    self._stamps.append(now)
                    return
                wait = 60 - (now - self._stamps[0]) + 0.05
            logger.info("RPM limit reached, sleeping %.1fs", wait)
            time.sleep(wait)


def _limiter_for(model: str, cfg: dict) -> _RpmLimiter:
    with _lock:
        if model not in _limiters:
            rpm = cfg["llm"].get("rpm", {}).get(model, 8)
            _limiters[model] = _RpmLimiter(rpm)
        return _limiters[model]


def _model_for_role(role: str, cfg: dict) -> str:
    return cfg["llm"][f"{role}_model"]


def _cache_dir(cfg: dict) -> Path:
    d = ROOT / cfg["llm"].get("cache_dir", ".cache/llm")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- LangChain path


def get_chat_model(role: str = "agent", *, streaming: bool = False, config: dict | None = None):
    """Rate-limited, cached ChatGoogleGenerativeAI for the given role
    (agent | guard | judge). Reused per (role, streaming)."""
    global _sqlite_cache_set
    cfg = config or load_config("agent")
    model = _model_for_role(role, cfg)

    key = (model, streaming)
    with _lock:
        if key in _chat_models:
            return _chat_models[key]

    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not _sqlite_cache_set and not streaming:
        # Streaming responses aren't cached by LangChain; only set the global
        # cache once, for non-streaming calls (judges, guards, evals).
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache

        set_llm_cache(SQLiteCache(database_path=str(_cache_dir(cfg) / "langchain.db")))
        _sqlite_cache_set = True

    rpm = cfg["llm"].get("rpm", {}).get(model, 8)
    chat = ChatGoogleGenerativeAI(
        model=model,
        temperature=cfg["llm"].get("temperature", 0.6),
        max_retries=cfg["llm"].get("max_retries", 4),
        rate_limiter=InMemoryRateLimiter(requests_per_second=rpm / 60, max_bucket_size=1),
        disable_streaming=not streaming,
    )
    with _lock:
        _chat_models[key] = chat
    return chat


# ---------------------------------------------------------------- plain-text path


def _message_text(message) -> str:
    """Message text on any langchain-core / Gemini pairing: content is a plain
    string on 2.5 models but a content-block list (text + thought signatures)
    on Gemini 3 — .content's repr would poison the guard/judge JSON parsing."""
    if isinstance(message.content, str):
        return message.content
    text = message.text
    return text if isinstance(text, str) else str(text())


def generate(
    prompt: str,
    role: str = "guard",
    *,
    config: dict | None = None,
    _call=None,  # test seam: inject a fake completion function
) -> str:
    """Plain text completion with file cache + RPM limit + backoff.

    Used by guards and judges where we want a single cheap call and an
    inspectable cache (one JSON file per request under .cache/llm/)."""
    cfg = config or load_config("agent")
    model = _model_for_role(role, cfg)

    digest = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:32]
    cache_file = _cache_dir(cfg) / f"{digest}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["response"]

    if _call is None:
        chat = get_chat_model(role, config=cfg)
        _call = lambda p: _message_text(chat.invoke(p))  # noqa: E731

    _limiter_for(model, cfg).acquire()

    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(cfg["llm"].get("max_retries", 4)):
        try:
            text = _call(prompt)
            cache_file.write_text(
                json.dumps({"model": model, "prompt": prompt, "response": text}, indent=2),
                encoding="utf-8",
            )
            return text
        except Exception as e:  # 429s / transient 5xx
            last_err = e
            logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err
