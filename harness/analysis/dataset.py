"""Loads a directory of negotiation transcripts into one row per negotiation.

Unlike the paper notebook, which asserted on every check (the Gemini data was
clean), problems are recorded as per-transcript flags. Open models break the
contract in more ways (wrong language, null prices on accept, prices outside
the targets), and the analysis should report those rates rather than crash on
the first one.
"""
from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from harness.analysis.design import (
    ALLOWED_ACTIONS,
    CONDITIONS,
    MAX_TURNS,
    N_REPETITIONS,
    PRICE_ACTIONS,
    ROLES,
    VALID_OUTCOMES,
)

TRANSCRIPT_NAME = re.compile(
    r"^(scenario-\d{2})_(en|ko)-(none|strong)-(cooperative|neutral|headstrong)_rep(\d+)\.json$"
)
HANGUL_RE = re.compile(r"[가-힣]")
EPSILON = 1e-12


def load_scenarios(path: Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for index, scenario in enumerate(payload):
        span = float(scenario["seller_target"]) - float(scenario["buyer_target"])
        if span <= 0:
            raise ValueError(f"scenario-{index:02d} has a non-positive bargaining span")
        rows.append({"scenario_id": f"scenario-{index:02d}", **scenario, "price_span": span})
    return pd.DataFrame(rows)


@dataclass
class TranscriptSet:
    """Everything loaded from one transcript directory."""

    attempts: pd.DataFrame  # one row per file on disk, valid or not
    transcripts: pd.DataFrame  # one row per analyzable negotiation, with metrics
    missing: list[str] = field(default_factory=list)  # design keys with no file


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def script_language(utterances: list[str]) -> str:
    """'ko', 'en' or 'mixed', by whether utterances contain Hangul."""
    has_hangul = [bool(HANGUL_RE.search(text)) for text in utterances]
    if all(has_hangul):
        return "ko"
    if not any(has_hangul):
        return "en"
    return "mixed"


ECHO_SIMILARITY = 0.8


def echo_turns(utterances: list[str]) -> int:
    """Turns that mostly repeat the previous turn's text (a degenerate copy loop).

    Small models sometimes copy the partner's last line instead of replying;
    such turns carry no negotiation behavior and are reported as a quality metric.
    """
    return sum(
        difflib.SequenceMatcher(None, current, previous).ratio() > ECHO_SIMILARITY
        for previous, current in zip(utterances, utterances[1:])
    )


def offer_metrics(turns: list[dict], role: str, buyer_target: float, span: float) -> dict:
    """Concession statistics on one role's price-bearing turns, in target-span units.

    Normalized price z = (p - buyer_target) / (seller_target - buyer_target), so a
    buyer conceding means z goes up and a seller conceding means z goes down.
    """
    prefix = f"{role}_"
    offers = [
        (index, (float(turn["price"]) - buyer_target) / span)
        for index, turn in enumerate(turns)
        if turn["role"] == role and turn.get("action") in PRICE_ACTIONS and _is_number(turn.get("price"))
    ]
    keys = [
        "offer_count", "first_offer_norm", "last_offer_norm", "concession_opportunities",
        "concession_count", "concession_rate", "gross_concession_norm", "net_concession_norm",
        "hardening_count", "hardening_norm", "mean_concession_norm",
        "first_concession_turn", "first_concession_progress",
    ]
    if not offers:
        # e.g. a one-turn acceptance: the buyer never made an offer. Kept as NaN so
        # outcome metrics still count this run while offer metrics skip it.
        return {prefix + key: (0 if key == "offer_count" else np.nan) for key in keys}

    values = np.array([value for _, value in offers])
    direction = 1.0 if role == "buyer" else -1.0
    steps = direction * np.diff(values)
    conceding = steps > EPSILON
    hardening = steps < -EPSILON
    opportunities = len(values) - 1

    if conceding.any():
        first_turn = offers[int(np.flatnonzero(conceding)[0]) + 1][0] + 1
        first_progress = (first_turn - 1) / max(len(turns) - 1, 1)
    else:
        first_turn, first_progress = np.nan, np.nan

    return {
        prefix + "offer_count": len(values),
        prefix + "first_offer_norm": float(values[0]),
        prefix + "last_offer_norm": float(values[-1]),
        prefix + "concession_opportunities": opportunities,
        prefix + "concession_count": int(conceding.sum()),
        prefix + "concession_rate": float(conceding.mean()) if opportunities else np.nan,
        prefix + "gross_concession_norm": float(np.clip(steps, 0, None).sum()),
        prefix + "net_concession_norm": float(direction * (values[-1] - values[0])),
        prefix + "hardening_count": int(hardening.sum()),
        prefix + "hardening_norm": float(np.clip(-steps, 0, None).sum()),
        prefix + "mean_concession_norm": float(steps[conceding].mean()) if conceding.any() else np.nan,
        prefix + "first_concession_turn": first_turn,
        prefix + "first_concession_progress": first_progress,
    }


def contract_issues(turns: list[dict], outcome: str, final_price) -> list[str]:
    """Departures from the negotiation output contract (schema, roles, prices)."""
    issues = []
    if not 1 <= len(turns) <= MAX_TURNS:
        issues.append("turn_count")
    for index, turn in enumerate(turns):
        if turn.get("role") != ("seller" if index % 2 == 0 else "buyer"):
            issues.append("role_alternation")
        action = turn.get("action")
        if action not in ALLOWED_ACTIONS:
            issues.append("unknown_action")
        if action in PRICE_ACTIONS and not (_is_number(turn.get("price")) and turn["price"] > 0):
            issues.append("offer_without_price")
        if action == "accept" and turn.get("price") is None:
            issues.append("accept_without_price")
        if action == "reject" and turn.get("price") is not None:
            issues.append("reject_with_price")
    if outcome == "agreed" and not _is_number(final_price):
        issues.append("agreement_without_price")
    return sorted(set(issues))


def load_transcripts(transcripts_dir: Path, scenarios: pd.DataFrame) -> TranscriptSet:
    scenario_lookup = scenarios.set_index("scenario_id").to_dict("index")
    expected = {
        f"{scenario_id}_{language}-{urgency}-{persona}_rep{rep}"
        for scenario_id in scenario_lookup
        for language, urgency, persona in CONDITIONS
        for rep in range(N_REPETITIONS)
    }

    attempts, rows, seen = [], [], set()
    for path in sorted(Path(transcripts_dir).glob("*.json")):
        match = TRANSCRIPT_NAME.fullmatch(path.name)
        if match is None:
            continue
        scenario_id, language, urgency, persona, rep = match.groups()
        rep = int(rep)
        transcript_id = path.stem
        seen.add(transcript_id)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        outcome = payload.get("outcome")
        base = {
            "transcript_id": transcript_id,
            "scenario_id": scenario_id,
            "language": language,
            "urgency": urgency,
            "persona": persona,
            "repetition": rep,
            "outcome": outcome,
        }
        if payload.get("condition_id") != f"{language}-{urgency}-{persona}" or payload.get("repetition") != rep:
            raise ValueError(f"Filename/payload mismatch in {path.name}")
        if outcome not in VALID_OUTCOMES:
            # parse_error: the negotiation never finished, so it is excluded from
            # every metric (the paper's two Gemini exclusions).
            attempts.append({**base, "status": "excluded", "reason": outcome})
            continue

        turns = payload["turns"]
        scenario = scenario_lookup[scenario_id]
        buyer_target, span = float(scenario["buyer_target"]), float(scenario["price_span"])
        final_price = payload.get("final_price")
        utterances = [turn.get("utterance", "") for turn in turns]
        spoken = script_language(utterances)
        issues = contract_issues(turns, outcome, final_price)

        row = {
            **base,
            "category": scenario["category"],
            "condition_id": f"{language}-{urgency}-{persona}",
            "n_turns": len(turns),
            "agreed": int(outcome == "agreed"),
            "abandoned": int(outcome == "abandoned"),
            "max_turn": int(outcome == "max_turns_reached"),
            "final_price": float(final_price) if _is_number(final_price) else np.nan,
            "normalized_final_price": (
                (float(final_price) - buyer_target) / span
                if outcome == "agreed" and _is_number(final_price)
                else np.nan
            ),
            "spoken_language": spoken,
            "language_compliant": spoken == language,
            "echo_turns": echo_turns(utterances),
            "first_turn_action": turns[0].get("action"),
            "contract_issues": ",".join(issues),
        }
        for role in ROLES:
            row.update(offer_metrics(turns, role, buyer_target, span))
        row["terminal_gap_norm"] = row["seller_last_offer_norm"] - row["buyer_last_offer_norm"]
        rows.append(row)
        attempts.append({**base, "status": "valid", "reason": None})

    return TranscriptSet(
        attempts=pd.DataFrame(attempts),
        transcripts=pd.DataFrame(rows).sort_values("transcript_id").reset_index(drop=True),
        missing=sorted(expected - seen),
    )
