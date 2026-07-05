import json

from src.guardrails.output_guard import check_output

CHUNKS = ["Rababb interned at BMO on the data platform team."]


def _judge(grounded, claims=()):
    return lambda prompt: json.dumps({"grounded": grounded, "ungrounded_claims": list(claims)})


def test_pii_blocks_before_judge_runs():
    def exploding_judge(prompt):
        raise AssertionError("judge should not run when PII found")

    decision = check_output("Sure, call me at 519-555-1234.", CHUNKS, judge=exploding_judge)
    assert not decision.ok
    assert decision.pii_hits == ["phone"]
    assert "private data" in decision.feedback()


def test_grounded_answer_passes():
    decision = check_output("I interned at BMO.", CHUNKS, judge=_judge(True))
    assert decision.ok


def test_ungrounded_answer_produces_feedback():
    decision = check_output(
        "I interned at BMO and also won a Nobel prize.",
        CHUNKS,
        judge=_judge(False, ["won a Nobel prize"]),
    )
    assert not decision.ok
    assert "Nobel" in decision.feedback()


def test_judge_error_fails_closed():
    def broken(prompt):
        raise RuntimeError("quota exhausted")

    decision = check_output("I interned at BMO.", CHUNKS, judge=broken)
    assert not decision.ok


def test_groundedness_skippable():
    decision = check_output("Anything at all.", [], judge=None, groundedness=False)
    assert decision.ok
