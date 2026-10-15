import numpy as np
import pandas as pd
import pytest

from harness.analysis.dataset import contract_issues, offer_metrics, script_language
from harness.analysis.design import CONDITIONS
from harness.analysis.stats import (
    CONTRASTS,
    bootstrap_contrast,
    exact_sign_flip_pvalue,
    holm_adjust,
)


@pytest.mark.parametrize("name", CONTRASTS)
def test_contrast_weights_sum_to_zero(name):
    assert abs(CONTRASTS[name].sum()) < 1e-12


def test_language_contrast_is_difference_of_language_means():
    cell_means = np.arange(len(CONDITIONS), dtype=float)
    ko = [m for m, c in zip(cell_means, CONDITIONS) if c[0] == "ko"]
    en = [m for m, c in zip(cell_means, CONDITIONS) if c[0] == "en"]
    assert cell_means @ CONTRASTS["language_ko_minus_en"] == pytest.approx(np.mean(ko) - np.mean(en))


def test_sign_flip_all_positive_effects():
    # With 12 identical positive effects only the all-+ and all-- flips are as extreme.
    assert exact_sign_flip_pvalue(np.ones(12)) == pytest.approx(2 / 2**12)


def test_sign_flip_zero_effect_is_one():
    assert exact_sign_flip_pvalue(np.zeros(12)) == 1.0


def test_holm_matches_textbook_example():
    adjusted = holm_adjust(pd.Series([0.01, 0.04, 0.03, 0.005]))
    assert adjusted.round(4).tolist() == [0.03, 0.06, 0.06, 0.02]


def test_bootstrap_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(0)
    matrix = rng.random((12, len(CONDITIONS)))
    first = bootstrap_contrast(matrix, CONTRASTS["language_ko_minus_en"], "unit", "test")
    second = bootstrap_contrast(matrix, CONTRASTS["language_ko_minus_en"], "unit", "test")
    assert first == second
    estimate, low, high, complete = first
    assert low <= estimate <= high
    assert complete == 12


def _turn(role, action, price=None):
    return {"role": role, "action": action, "price": price, "utterance": "x"}


def test_offer_metrics_buyer_concession():
    # targets: buyer 100, seller 200 -> z = (p - 100) / 100
    turns = [
        _turn("seller", "offer", 200),
        _turn("buyer", "counter-offer", 110),
        _turn("seller", "counter-offer", 180),
        _turn("buyer", "counter-offer", 130),
        _turn("seller", "accept", 130),
    ]
    buyer = offer_metrics(turns, "buyer", buyer_target=100, span=100)
    seller = offer_metrics(turns, "seller", buyer_target=100, span=100)
    assert buyer["buyer_first_offer_norm"] == pytest.approx(0.1)
    assert buyer["buyer_net_concession_norm"] == pytest.approx(0.2)
    assert buyer["buyer_first_concession_turn"] == 4
    assert seller["seller_net_concession_norm"] == pytest.approx(0.2)


def test_offer_metrics_without_offers_is_nan():
    turns = [_turn("seller", "offer", 200), _turn("buyer", "accept", 200)]
    buyer = offer_metrics(turns, "buyer", buyer_target=100, span=100)
    assert buyer["buyer_offer_count"] == 0
    assert np.isnan(buyer["buyer_first_offer_norm"])


def test_script_language():
    assert script_language(["안녕하세요", "좋아요 $100"]) == "ko"
    assert script_language(["hello", "deal"]) == "en"
    assert script_language(["hello", "좋아요"]) == "mixed"


def test_contract_issues_flags_open_model_quirks():
    turns = [_turn("seller", "offer", 200), _turn("buyer", "accept", None)]
    assert contract_issues(turns, "agreed", 200) == ["accept_without_price"]
    turns = [_turn("seller", "offer", 200), _turn("buyer", "reject", 150)]
    assert contract_issues(turns, "abandoned", None) == ["reject_with_price"]


def test_echo_turns_counts_copied_lines():
    from harness.analysis.dataset import echo_turns
    assert echo_turns(["I offer $200 for the bike", "I offer $200 for the bike!", "No, $150"]) == 1
    assert echo_turns(["hello"]) == 0
