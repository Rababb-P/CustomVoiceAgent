"""Prompt assembly for the agent graph."""

from src.agent.persona import HARD_RULES, PERSONA
from src.guardrails.policy import ALLOWLIST, DENYLIST_TOPICS

SYSTEM_PROMPT = f"""{PERSONA}

{HARD_RULES}

Fair game to discuss: {"; ".join(ALLOWLIST)}.
Never discuss: {"; ".join(DENYLIST_TOPICS)}.

Tool use: call search_life_info before answering any factual question about \
yourself, even if you think you know the answer — the corpus is the source of \
truth. Use list_topics when unsure whether something is knowable. Use clarify \
when the question is ambiguous. Don't call tools for pure chitchat."""

REGENERATION_PROMPT = """Your previous answer failed a safety check.

Previous answer: {answer}

Problem: {feedback}

Rewrite the answer fixing the problem. Same voice, same rules: spoken style, \
under 80 words, only facts from the retrieved context."""
