"""End-to-end answer quality: LLM-as-judge over the full guarded pipeline.

Fixture: data/evals/agent_eval.jsonl —
{"question": ..., "gold_facts": [...], "rubric_notes": "optional per-question guidance"}

The judge sees gold FACTS, never gold answers, and scores 1-5 on four axes.
Judge calls go through src.llm.generate, so they're cached by (model, prompt
hash) — reruns on unchanged answers cost zero quota.
"""

from __future__ import annotations

RUBRIC = """Score the answer 1-5 on each axis (5 = excellent):
- faithfulness: every factual claim is consistent with the gold facts; no invention.
  An honest "I don't have that info" when gold facts are empty scores 5.
- persona: sounds like a direct, casual engineering student speaking in first person.
- tts_friendliness: short spoken-style sentences; no markdown, lists, URLs, or emoji.
- directness: answers the actual question quickly, no filler or hedging."""

_JUDGE_PROMPT = """You are grading a voice agent that answers as Rababb Pannu.

{rubric}
{notes}
Question: {question}

Gold facts (source of truth, may be empty):
{facts}

Agent's answer:
{answer}

Reply with exactly one JSON object, no prose:
{{"faithfulness": n, "persona": n, "tts_friendliness": n, "directness": n, "comment": "..."}}"""


def run() -> dict:
    from evals.common import load_jsonl, parse_judge_json
    from src.agent.graph import ask, build_graph
    from src.llm import generate

    rows = load_jsonl("agent_eval.jsonl")
    if not rows:
        return {"skipped": "no agent_eval.jsonl"}

    graph = build_graph()
    axes = ["faithfulness", "persona", "tts_friendliness", "directness"]
    totals = dict.fromkeys(axes, 0.0)
    per_question = []

    for i, row in enumerate(rows):
        turn = ask(graph, row["question"], thread_id=f"eval-{i}")  # fresh thread per question
        notes = f"Extra guidance: {row['rubric_notes']}\n" if row.get("rubric_notes") else ""
        raw = generate(
            _JUDGE_PROMPT.format(
                rubric=RUBRIC,
                notes=notes,
                question=row["question"],
                facts="\n".join(f"- {f}" for f in row.get("gold_facts", [])) or "(none)",
                answer=turn["answer"],
            ),
            role="judge",
        )
        try:
            scores = parse_judge_json(raw)
        except Exception:
            scores = dict.fromkeys(axes, 0)
            scores["comment"] = f"judge parse failure: {raw[:100]}"
        for a in axes:
            totals[a] += float(scores.get(a, 0))
        per_question.append({"question": row["question"], "answer": turn["answer"], **scores})

    n = len(rows)
    means = {a: round(totals[a] / n, 2) for a in axes}
    return {
        "n": n,
        "mean_scores": means,
        "mean_overall": round(sum(means.values()) / len(axes), 2),
        "per_question": per_question,
    }


def main() -> None:
    import json

    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
