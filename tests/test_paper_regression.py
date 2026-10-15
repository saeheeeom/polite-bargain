"""The ported analysis must reproduce the published numbers from the original transcripts.

Expected values are copied from the camera-ready paper (paper/main.tex): Table 1
and Section 3.2 for Gemini, Appendix D (Table 4) for Qwen. Transcripts are not
in git, so these tests skip wherever results/ is absent (e.g. CI).
"""
from pathlib import Path

import pytest

from harness.analysis.dataset import load_scenarios, load_transcripts
from harness.analysis.stats import factorial_contrasts

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "data" / "scenarios.json"
GEMINI_DIR = ROOT / "results" / "transcripts"
QWEN_DIR = ROOT / "results" / "transcripts_qwen"

pytestmark = pytest.mark.paper_data


def _contrasts(transcripts_dir: Path):
    if not (SCENARIOS.is_file() and any(transcripts_dir.glob("*.json"))):
        pytest.skip(f"original transcripts not available under {transcripts_dir}")
    scenarios = load_scenarios(SCENARIOS)
    loaded = load_transcripts(transcripts_dir, scenarios)
    table = factorial_contrasts(loaded.transcripts, scenarios["scenario_id"].tolist())
    return loaded, table.set_index(["metric", "contrast"])


@pytest.fixture(scope="module")
def gemini():
    return _contrasts(GEMINI_DIR)


@pytest.fixture(scope="module")
def qwen():
    return _contrasts(QWEN_DIR)


def test_gemini_counts(gemini):
    loaded, _ = gemini
    t = loaded.transcripts
    assert loaded.missing == []
    assert len(loaded.attempts) == 432
    assert sorted(loaded.attempts.query("status == 'excluded'")["transcript_id"]) == [
        "scenario-00_en-strong-headstrong_rep1",
        "scenario-06_en-strong-headstrong_rep0",
    ]
    assert len(t) == 430
    assert int(t["n_turns"].sum()) == 4580
    assert t["outcome"].value_counts().to_dict() == {"agreed": 279, "abandoned": 118, "max_turns_reached": 33}
    assert t["language_compliant"].all()


# (metric, contrast): (estimate, ci_low, ci_high, holm_p or None), rounded as printed in the paper.
GEMINI_TABLE = {
    ("agreed", "language_ko_minus_en"): (24.8, 15.7, 34.7, 0.0039),
    ("agreed", "urgency_strong_minus_none"): (29.9, 20.1, 39.6, None),
    ("agreed", "language_x_urgency"): (16.2, 0.9, 32.9, 0.1133),
    ("agreed", "language_x_cooperative_vs_neutral"): (-48.6, -66.7, -30.6, 0.0039),
    ("agreed", "language_x_headstrong_vs_neutral"): (-22.9, -36.8, -8.3, 0.0391),
}
GEMINI_PROCESS = {
    ("buyer_first_offer_norm", "language_ko_minus_en"): (0.024, -0.011, 0.056),
    ("buyer_net_concession_norm", "language_ko_minus_en"): (-0.086, -0.137, -0.041),
    ("seller_net_concession_norm", "language_ko_minus_en"): (0.349, 0.237, 0.462),
    ("terminal_gap_norm", "language_ko_minus_en"): (-0.286, -0.400, -0.175),
    ("normalized_final_price", "language_ko_minus_en"): (-0.250, -0.315, -0.178),
}
GEMINI_TURNS = {
    ("n_turns", "language_ko_minus_en"): (0.41, -0.17, 1.07),
    ("n_turns", "urgency_strong_minus_none"): (-0.94, -1.58, -0.33),
}


@pytest.mark.parametrize("key", GEMINI_TABLE)
def test_gemini_headline_table(gemini, key):
    _, table = gemini
    row = table.loc[key]
    estimate, low, high, holm = GEMINI_TABLE[key]
    assert round(100 * row["estimate"], 1) == estimate
    assert round(100 * row["ci_low"], 1) == low
    assert round(100 * row["ci_high"], 1) == high
    if holm is not None:
        assert round(row["holm_p"], 4) == holm


@pytest.mark.parametrize("key", GEMINI_PROCESS)
def test_gemini_process_measures(gemini, key):
    _, table = gemini
    row = table.loc[key]
    assert tuple(round(row[c], 3) for c in ("estimate", "ci_low", "ci_high")) == GEMINI_PROCESS[key]


@pytest.mark.parametrize("key", GEMINI_TURNS)
def test_gemini_turns(gemini, key):
    _, table = gemini
    row = table.loc[key]
    assert tuple(round(row[c], 2) for c in ("estimate", "ci_low", "ci_high")) == GEMINI_TURNS[key]


def test_qwen_counts(qwen):
    loaded, _ = qwen
    t = loaded.transcripts
    assert len(t) == 432
    assert t["outcome"].value_counts().to_dict() == {"abandoned": 237, "agreed": 180, "max_turns_reached": 15}
    assert int((~t["language_compliant"]).sum()) == 12  # Korean-condition runs spoken in English
    assert int(t["buyer_first_offer_norm"].notna().sum()) == 425  # 7 one-turn acceptances


# Point estimates only: the paper's Qwen intervals came from the notebook's
# per-language marginal summaries, which use different bootstrap seed labels.
QWEN_ESTIMATES = {
    ("agreed", "language_ko_minus_en"): 0.139,
    ("agreed", "urgency_strong_minus_none"): 0.444,
    ("buyer_first_offer_norm", "language_ko_minus_en"): 0.017,
    ("buyer_net_concession_norm", "language_ko_minus_en"): 0.009,
    ("seller_net_concession_norm", "language_ko_minus_en"): 0.121,
    ("terminal_gap_norm", "language_ko_minus_en"): -0.147,
}


@pytest.mark.parametrize("key", QWEN_ESTIMATES)
def test_qwen_estimates(qwen, key):
    _, table = qwen
    assert round(table.loc[key, "estimate"], 3) == QWEN_ESTIMATES[key]
