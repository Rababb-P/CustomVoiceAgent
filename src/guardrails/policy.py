"""PII policy: what the agent may and may not disclose.

Allowlist is documentation for the persona prompt; the denylist is enforced —
regex here, plus a judge check in output_guard for things regex can't catch
(e.g. facts about family/friends phrased without obvious patterns).
"""

from __future__ import annotations

import re

# Public-by-choice facts. Referenced by prompts.py so the persona knows what's fair game.
ALLOWLIST = [
    "school, program, and graduation timeline",
    "roles, employers, and public project work",
    "hackathon wins and public awards",
    "public email address",
    "interests and hobbies",
]

# Hard denials, regardless of what's in the corpus.
DENYLIST_TOPICS = [
    "phone numbers, home or mailing addresses",
    "government IDs (SIN, passport, driver's licence)",
    "banking or financial details",
    "health information",
    "anything about family members or friends (third parties)",
    "credentials, API keys, passwords",
]

# Pattern name -> compiled regex over the *output* text.
_PATTERNS: dict[str, re.Pattern] = {
    "phone": re.compile(r"\+?1?[\s.-]?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
    "sin": re.compile(r"\b\d{3}[\s-]\d{3}[\s-]\d{3}\b"),
    "street_address": re.compile(
        r"\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr|blvd|court|ct|lane|ln|crescent|cres)\b",
        re.IGNORECASE,
    ),
    "postal_code": re.compile(r"\b[A-Za-z]\d[A-Za-z][\s-]?\d[A-Za-z]\d\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "api_key": re.compile(r"\b(sk-|AIza|ghp_|AKIA)[A-Za-z0-9_-]{10,}"),
}

PUBLIC_EMAIL = "rpannu@uwaterloo.ca"
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def find_pii(text: str) -> list[str]:
    """Return the names of denylist patterns present in text (empty = clean)."""
    hits = [name for name, pat in _PATTERNS.items() if pat.search(text)]
    for email in _EMAIL.findall(text):
        if email.lower() != PUBLIC_EMAIL:
            hits.append("non_public_email")
            break
    return hits
