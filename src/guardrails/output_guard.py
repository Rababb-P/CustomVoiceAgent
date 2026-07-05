"""Output guard: PII denylist scan + groundedness judge.

Pure function with the judge injected as a callable. The graph gives failed
answers one regeneration attempt (with the violation as feedback) before
falling back to a safe response.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from src.guardrails.policy import find_pii

logger = logging.getLogger(__name__)


@dataclass
class OutputDecision:
    ok: bool
    reason: str
    ungrounded_claims: list[str] = field(default_factory=list)
    pii_hits: list[str] = field(default_factory=list)

    def feedback(self) -> str:
        """Feedback injected into the regeneration prompt."""
        parts = []
        if self.pii_hits:
            parts.append(f"Remove all private data ({', '.join(self.pii_hits)}).")
        if self.ungrounded_claims:
            claims = "; ".join(self.ungrounded_claims)
            parts.append(
                "These claims are not supported by the retrieved context, "
                f"drop or soften them: {claims}"
            )
        return " ".join(parts)


_JUDGE_PROMPT = """You are checking whether an answer only states facts supported by \
the provided context. Opinions, greetings, and refusals need no support — only \
concrete factual claims about Rababb's life, work, or projects do.

Context chunks:
{chunks}

Answer to check:
{answer}

Reply with exactly one JSON object, no prose:
{{"grounded": true|false, "ungrounded_claims": ["<claim>", ...]}}"""

SAFE_FALLBACK = (
    "Honestly, I'm not sure about that one — it's not something I have solid info on. "
    "Ask me about my projects or work and I've got you."
)


def check_output(
    answer: str,
    chunks: list[str],
    *,
    judge: Callable[[str], str] | None = None,
    groundedness: bool = True,
) -> OutputDecision:
    pii = find_pii(answer)
    if pii:
        logger.warning("output guard: PII hit %s", pii)
        return OutputDecision(False, f"PII denylist hit: {pii}", pii_hits=pii)

    if not groundedness or judge is None:
        return OutputDecision(True, "pii clean (groundedness skipped)")

    context = "\n---\n".join(chunks) if chunks else "(no chunks retrieved this turn)"
    try:
        raw = judge(_JUDGE_PROMPT.format(chunks=context, answer=answer))
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except Exception as e:
        # Fail closed on judge errors: an unverifiable answer is treated as ungrounded.
        logger.warning("groundedness judge failed (%s); failing closed", e)
        return OutputDecision(False, f"judge error: {e}", ungrounded_claims=["<judge unavailable>"])

    if data.get("grounded", False):
        return OutputDecision(True, "grounded, pii clean")
    claims = [str(c) for c in data.get("ungrounded_claims", [])]
    logger.info("output guard: ungrounded claims %s", claims)
    return OutputDecision(False, "ungrounded claims", ungrounded_claims=claims)
