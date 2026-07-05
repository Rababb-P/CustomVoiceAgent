"""Cut a token stream into speakable chunks at sentence boundaries.

First audio should play while the LLM is still generating, so we flush as soon
as a sentence completes. Long run-on sentences are flushed at clause boundaries
(comma/semicolon/dash) past a soft length limit, and unconditionally past a hard
limit, so latency never depends on the model producing a period.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable, Iterator

_SENTENCE_END = re.compile(r"[.!?]['\")\]]?\s")
_CLAUSE_END = re.compile(r"[,;:—]\s")

SOFT_LIMIT = 120   # chars: prefer clause breaks past this
HARD_LIMIT = 240   # chars: flush no matter what


def _split_ready(buffer: str) -> tuple[list[str], str]:
    """Split off every complete sentence; apply clause/hard limits to the tail."""
    ready: list[str] = []
    while True:
        m = _SENTENCE_END.search(buffer)
        if m:
            ready.append(buffer[: m.end()].strip())
            buffer = buffer[m.end():]
            continue
        if len(buffer) > SOFT_LIMIT:
            c = _CLAUSE_END.search(buffer, SOFT_LIMIT // 2)
            if c:
                ready.append(buffer[: c.end()].strip())
                buffer = buffer[c.end():]
                continue
        if len(buffer) > HARD_LIMIT:
            cut = buffer.rfind(" ", 0, HARD_LIMIT)
            cut = cut if cut > 0 else HARD_LIMIT
            ready.append(buffer[:cut].strip())
            buffer = buffer[cut:]
            continue
        return [r for r in ready if r], buffer


def chunk_stream(tokens: Iterable[str]) -> Iterator[str]:
    """Synchronous version, used by tests and the POST /ask path."""
    buffer = ""
    for token in tokens:
        buffer += token
        ready, buffer = _split_ready(buffer)
        yield from ready
    tail = buffer.strip()
    if tail:
        yield tail


async def achunk_stream(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Async version for the WebSocket pipeline (LangGraph token stream in)."""
    buffer = ""
    async for token in tokens:
        buffer += token
        ready, buffer = _split_ready(buffer)
        for r in ready:
            yield r
    tail = buffer.strip()
    if tail:
        yield tail
