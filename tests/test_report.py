from evals.report import check_regressions


def test_clean_run_passes():
    current = {
        "rag": {"recall@6": 0.92, "mrr": 0.8},
        "safety": {"pass_rate_by_category": {"injection": 1.0, "pii_fishing": 1.0}},
        "agent": {"mean_overall": 4.2},
        "asr": {"finetuned": {"wer": 0.10}},
    }
    previous = {
        "agent": {"mean_overall": 4.1},
        "asr": {"finetuned": {"wer": 0.10}},
    }
    assert check_regressions(current, previous) == []


def test_recall_floor():
    assert check_regressions({"rag": {"recall@6": 0.7}}, None)


def test_safety_floor_is_absolute():
    current = {"safety": {"pass_rate_by_category": {"injection": 0.9, "pii_fishing": 1.0}}}
    problems = check_regressions(current, None)
    assert any("injection" in p for p in problems)


def test_wer_relative_regression():
    current = {"asr": {"finetuned": {"wer": 0.12}}}
    previous = {"asr": {"finetuned": {"wer": 0.10}}}
    assert any("WER" in p for p in check_regressions(current, previous))


def test_judge_score_drop():
    current = {"agent": {"mean_overall": 3.5}}
    previous = {"agent": {"mean_overall": 4.0}}
    assert any("judge" in p for p in check_regressions(current, previous))


def test_first_run_no_baseline():
    current = {"rag": {"recall@6": 0.95}}
    assert check_regressions(current, None) == []
