"""
One-off script that appends a Qwen3-negotiator generalizability-check section
to notebooks/full_experiment_analysis.ipynb and re-executes the whole notebook
top-to-bottom, then saves it back in place.

Reuses the existing notebook's own validation logic and statistics functions
(scenario_balanced_summary, condition_balanced_marginal_summary,
bootstrap_mean_interval, stable_seed) rather than reimplementing them, so the
Gemini-vs-Qwen comparison is computed with an identical methodology. Does NOT
touch FTA/politeness pragmatics — the Qwen batch has no judged/*.json files by
design (see design_changes_log.md: judge model reuse would self-evaluate).
"""
import nbformat as nbf
from nbclient import NotebookClient

NB_PATH = "notebooks/full_experiment_analysis.ipynb"

nb = nbf.read(NB_PATH, as_version=4)

new_cells = []

new_cells.append(nbf.v4.new_markdown_cell(
"""## 12. Generalizability check — Qwen3 negotiator

`old_research_statement.md` (line 91) plans rerunning core negotiation
results on a second negotiator model, Qwen3, to check whether the language
effect found above is specific to Gemini or generalizes. This section loads
`results/transcripts_qwen/` (Qwen3 as both buyer and seller, same 12x12x3
design) and compares agreement rate and conditional final price by language
against the Gemini results already computed above.

This section deliberately does **not** redo FTA/politeness judging on the
Qwen batch: the current judge model (`qwen/qwen3.7-flash`) is the same model
family as this negotiator, which would reintroduce the self-evaluation-bias
problem the judge model was chosen to avoid (see `design_changes_log.md`).
The comparison here is outcome-level only (agreement, price, turns).

**Data-quality note:** unlike Gemini (100% language-instruction compliant),
Qwen3 sometimes ignores the "ko" condition's Korean-language instruction and
negotiates an entire transcript in English (never partially — a transcript is
either all-Korean or all-English). These are excluded below and reported
separately, alongside the small number of `parse_error` transcripts, rather
than counted as Korean-condition data."""
))

new_cells.append(nbf.v4.new_code_cell(
'''QWEN_TRANSCRIPTS_DIR = ROOT / "results" / "transcripts_qwen"
qwen_transcript_paths = sorted(QWEN_TRANSCRIPTS_DIR.glob("*.json"))
require(
    len(qwen_transcript_paths) == expected_attempts,
    f"Expected {expected_attempts} Qwen transcript files, found {len(qwen_transcript_paths)}",
)

qwen_transcripts_by_key = {}
for path in qwen_transcript_paths:
    match = TRANSCRIPT_NAME.fullmatch(path.name)
    require(match is not None, f"Unexpected Qwen transcript filename: {path.name}")
    payload = read_json(path)
    require(isinstance(payload, dict), f"Qwen transcript must be an object: {path.name}")
    key = key_from_payload(payload)
    filename_key = (match.group(1), match.group(2), match.group(3), match.group(4), int(match.group(5)))
    require(key == filename_key, f"Filename/payload mismatch in {path.name}")
    require(key not in qwen_transcripts_by_key, f"Duplicate Qwen transcript key: {key}")
    qwen_transcripts_by_key[key] = (path, payload)

require(set(qwen_transcripts_by_key) == expected_keys, "Qwen transcript keys do not match the factorial grid")

qwen_valid_records = []
qwen_exclusion_records = []

for key in sorted(qwen_transcripts_by_key):
    path, transcript = qwen_transcripts_by_key[key]
    scenario_id, language, urgency, persona, repetition = key
    scenario = scenario_lookup[scenario_id]
    condition_id = f"{language}-{urgency}-{persona}"
    transcript_id = f"{scenario_id}_{condition_id}_rep{repetition}"

    outcome = transcript.get("outcome")
    if outcome == "parse_error":
        qwen_exclusion_records.append(
            {"transcript_file": path.name, "condition_id": condition_id, "reason": "parse_error"}
        )
        continue
    require(outcome in VALID_OUTCOMES, f"Invalid outcome in {path.name}")

    turns = transcript.get("turns")
    require(isinstance(turns, list) and turns, f"Missing turns in {path.name}")
    require(transcript.get("n_turns") == len(turns), f"n_turns mismatch in {path.name}")
    require(1 <= len(turns) <= MAX_TURNS, f"Invalid turn count in {path.name}")
    require(turns[0].get("role") == "seller", f"Seller must open in {path.name}")

    utterances = []
    for turn_index, turn in enumerate(turns):
        expected_role = "seller" if turn_index % 2 == 0 else "buyer"
        require(turn.get("role") == expected_role, f"Role alternation failed in {path.name} turn {turn_index}")
        action = turn.get("action")
        require(action in ALLOWED_ACTIONS, f"Invalid action in {path.name} turn {turn_index}")
        utterance = turn.get("utterance")
        require(isinstance(utterance, str) and utterance.strip(), f"Invalid utterance in {path.name} turn {turn_index}")
        utterances.append(utterance)
        price = turn.get("price")
        if action in PRICE_ACTIONS:
            require(is_finite_number(price) and float(price) > 0, f"Invalid offer price in {path.name} turn {turn_index}")
        # Unlike Gemini, Qwen3 sometimes leaves price null on "accept" (rather than
        # restating the accepted price) and sometimes restates a price alongside
        # "reject" instead of nulling it. Both are real model-behavior variation,
        # not data bugs -- the transcript's top-level final_price (used for all
        # metrics here, not any turn-level price) is independently computed by
        # negotiation.py from the last non-null offer regardless, so per-turn price
        # presence isn't enforced on accept/reject the way the Gemini loader above
        # does.

    # Unlike Gemini (100% compliant), Qwen3 sometimes ignores the Korean-language
    # instruction entirely and negotiates the whole transcript in English despite
    # being in a "ko" condition (never partially -- it's all-Korean or all-English
    # per transcript). Including these as "ko" data would dilute the measured
    # language effect with transcripts that aren't actually in Korean, so they're
    # excluded and reported separately rather than silently kept or silently
    # dropped without a record.
    has_hangul = [bool(HANGUL_RE.search(text)) for text in utterances]
    if language == "ko" and not any(has_hangul):
        qwen_exclusion_records.append(
            {"transcript_file": path.name, "condition_id": condition_id, "reason": "qwen_language_noncompliance"}
        )
        continue
    require(
        all(has_hangul) if language == "ko" else not any(has_hangul),
        f"Mixed-language transcript (partial Korean compliance) in {path.name} -- unexpected, investigate",
    )

    final_price = transcript.get("final_price")
    if outcome == "agreed":
        require(turns[-1]["action"] == "accept", f"Agreement must end in accept in {path.name}")
        require(is_finite_number(final_price), f"Agreement must have final_price in {path.name}")
    elif outcome == "abandoned":
        require(turns[-1]["action"] == "reject", f"Abandonment must end in reject in {path.name}")
        require(final_price is None, f"Abandonment must have null final_price in {path.name}")
    else:
        require(len(turns) == MAX_TURNS, f"Turn-cap outcome must use {MAX_TURNS} turns in {path.name}")
        require(final_price is None, f"Turn-cap outcome must have null final_price in {path.name}")

    normalized_final_price = (
        (float(final_price) - float(scenario["buyer_target"])) / float(scenario["price_span"])
        if outcome == "agreed"
        else np.nan
    )

    qwen_valid_records.append({
        "transcript_id": transcript_id,
        "transcript_file": path.name,
        "scenario_id": scenario_id,
        "category": scenario["category"],
        "condition_id": condition_id,
        "language": language,
        "urgency": urgency,
        "persona": persona,
        "repetition": repetition,
        "outcome": outcome,
        "final_price": float(final_price) if final_price is not None else np.nan,
        "normalized_final_price": normalized_final_price,
        "n_turns": len(turns),
        "buyer_target": float(scenario["buyer_target"]),
        "seller_target": float(scenario["seller_target"]),
        "listing_price": float(scenario["listing_price"]),
        "price_span": float(scenario["price_span"]),
        "agreed": int(outcome == "agreed"),
        "abandoned": int(outcome == "abandoned"),
        "max_turn": int(outcome == "max_turns_reached"),
    })

qwen_transcript_df = pd.DataFrame(qwen_valid_records).sort_values(
    ["scenario_id", "language", "urgency", "persona", "repetition"]
).reset_index(drop=True)

require(
    transcript_df.loc[transcript_df["agreed"] == 1, "normalized_final_price"].between(0, 1).all(),
    "Agreed normalized prices must lie in the target interval",
)
# Qwen3, unlike Gemini, occasionally agrees slightly beyond a party's own target
# (e.g. buyer accepts $1 below their target -- a legitimately better-than-hoped
# deal, not a data bug). Report rather than hard-fail on out-of-range agreements.
qwen_out_of_range = qwen_transcript_df.loc[
    (qwen_transcript_df["agreed"] == 1) & ~qwen_transcript_df["normalized_final_price"].between(0, 1)
]
if len(qwen_out_of_range):
    print(f"Note: {len(qwen_out_of_range)} Qwen agreement(s) settled slightly beyond a party's target:")
    display(qwen_out_of_range[["transcript_id", "final_price", "buyer_target", "seller_target", "normalized_final_price"]])

qwen_exclusions_df = pd.DataFrame(qwen_exclusion_records)
require(
    len(qwen_transcript_df) + len(qwen_exclusions_df) == expected_attempts,
    "Qwen valid and excluded transcripts must partition the expected design grid",
)

print(f"Qwen3 negotiator dataset: {len(qwen_transcript_df)} valid transcripts")
if len(qwen_exclusions_df):
    print(f"Excluded {len(qwen_exclusions_df)}:")
    display(qwen_exclusions_df["reason"].value_counts())
    display(qwen_exclusions_df)
display(qwen_transcript_df["outcome"].value_counts())'''
))

new_cells.append(nbf.v4.new_markdown_cell(
"### 12.1 Scenario-balanced summaries for the Qwen3 negotiator"
))

new_cells.append(nbf.v4.new_code_cell(
'''qwen_primary_cell_long = scenario_balanced_summary(
    qwen_transcript_df, PRIMARY_METRICS, CONDITION_COLUMNS, label="qwen_primary_cell",
)

qwen_marginal_frames = []
for factor in CONDITION_COLUMNS:
    factor_summary = condition_balanced_marginal_summary(
        qwen_transcript_df, PRIMARY_METRICS, [factor], label=f"qwen_primary_marginal_{factor}",
    )
    factor_summary.insert(0, "factor", factor)
    factor_summary = factor_summary.rename(columns={factor: "level"})
    qwen_marginal_frames.append(factor_summary)
qwen_primary_marginal_summary = pd.concat(qwen_marginal_frames, ignore_index=True)

display(qwen_primary_marginal_summary.loc[qwen_primary_marginal_summary["factor"] == "language"])'''
))

new_cells.append(nbf.v4.new_markdown_cell(
"### 12.2 Gemini vs. Qwen3: does the language effect replicate?"
))

new_cells.append(nbf.v4.new_code_cell(
'''def marginal_by_language(marginal_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    subset = marginal_df.loc[
        (marginal_df["factor"] == "language")
        & (marginal_df["measure"].isin(["agreed", "normalized_final_price"]))
    ].copy()
    subset["model"] = model_name
    return subset[["model", "level", "measure", "estimate", "ci_low", "ci_high", "n_observations"]]

model_comparison = pd.concat(
    [
        marginal_by_language(primary_marginal_summary, "gemini-3.6-flash"),
        marginal_by_language(qwen_primary_marginal_summary, "qwen3.7-flash"),
    ],
    ignore_index=True,
).rename(columns={"level": "language"})
model_comparison["measure"] = model_comparison["measure"].map(
    {"agreed": "agreement_rate", "normalized_final_price": "normalized_final_price"}
)

display(model_comparison)

comparison_pivot = model_comparison.pivot_table(
    index=["model", "measure"], columns="language", values="estimate"
)
comparison_pivot["ko_minus_en"] = comparison_pivot["ko"] - comparison_pivot["en"]
display(comparison_pivot)'''
))

new_cells.append(nbf.v4.new_code_cell(
'''fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

for ax, measure, title, ylabel in zip(
    axes,
    ["agreement_rate", "normalized_final_price"],
    ["Agreement rate by language", "Conditional final price by language"],
    ["agreement rate", "normalized final price (0=buyer target, 1=seller target)"],
):
    plot_df = model_comparison.loc[model_comparison["measure"] == measure]
    for model_name, marker in zip(["gemini-3.6-flash", "qwen3.7-flash"], ["o", "s"]):
        sub = plot_df.loc[plot_df["model"] == model_name].set_index("language").loc[["en", "ko"]]
        ax.errorbar(
            sub.index,
            sub["estimate"],
            yerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
            marker=marker,
            capsize=4,
            label=model_name,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("language")

axes[0].legend()
fig.suptitle("Generalizability check: does the language effect hold with a second negotiator model?")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "qwen_generalizability_check.png")
plt.show()'''
))

new_cells.append(nbf.v4.new_code_cell(
'''qwen_transcript_df.to_csv(DATA_DIR / "qwen_transcript_metrics.csv", index=False)
model_comparison.to_csv(TABLE_DIR / "gemini_vs_qwen_language_effect.csv", index=False)

gemini_agree_gap = float(comparison_pivot.loc[("gemini-3.6-flash", "agreement_rate"), "ko_minus_en"])
qwen_agree_gap = float(comparison_pivot.loc[("qwen3.7-flash", "agreement_rate"), "ko_minus_en"])
same_direction = (gemini_agree_gap > 0) == (qwen_agree_gap > 0)

print(f"Gemini KO-EN agreement-rate gap: {gemini_agree_gap:+.3f}")
print(f"Qwen3  KO-EN agreement-rate gap: {qwen_agree_gap:+.3f}")
print(f"Same direction: {same_direction}")
print(
    "\\nNote: this is a directional check on point estimates, not a confirmatory "
    "cross-model significance test — no bootstrap/exact test compares the two "
    "models directly. Treat as descriptive evidence for the generalizability "
    "claim, not a third confirmatory hypothesis test."
)'''
))

nb["cells"] = list(nb["cells"]) + new_cells

client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
client.execute()

nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH} with {len(nb['cells'])} cells total")
