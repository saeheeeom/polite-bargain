"""
One-off generator for notebooks/smoke_test_analysis.ipynb.

Not part of the experiment pipeline — just builds the notebook via nbformat
and executes it so the checked-in .ipynb has real, current outputs. Re-run
this after the full 432-negotiation experiment to regenerate the notebook
against the full dataset (change TRANSCRIPTS_DIR below).
"""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Smoke test analysis

Analyzes the 36 negotiation transcripts in `results/transcripts/` from the
Gemini/Vertex AI smoke test (2026-08-15): **1 scenario** (`scenario-00`,
the mattress/box-spring/frame listing) x **12 conditions**
(language x urgency x persona) x **3 repetitions**.

**Caveat — read before drawing conclusions:** this is a pipeline sanity
check, not the real experiment. With only 1 scenario and n=3 per condition,
none of the breakdowns below have the sample size to say anything about the
actual research hypotheses (H1, H4 in research_statement.md). The point
here is to confirm the transcript schema, outcome coding, and price
extraction all look correct before running the full 432-negotiation
experiment across all 12 scenarios."""
))

cells.append(nbf.v4.new_code_cell(
"""import glob
import json

import matplotlib.pyplot as plt
import pandas as pd

TRANSCRIPTS_DIR = "../results/transcripts"

rows = []
for path in sorted(glob.glob(f"{TRANSCRIPTS_DIR}/*.json")):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    language, urgency, persona = d["condition_id"].split("-", 2)
    rows.append({
        "scenario_id": d["scenario_id"],
        "condition_id": d["condition_id"],
        "language": language,
        "urgency": urgency,
        "persona": persona,
        "repetition": d["repetition"],
        "outcome": d["outcome"],
        "final_price": d["final_price"],
        "n_turns": d["n_turns"],
    })

df = pd.DataFrame(rows)
print(f"{len(df)} transcripts loaded")
df.head()"""
))

cells.append(nbf.v4.new_markdown_cell("## Outcome distribution\n\nWhat fraction of negotiations converged vs. timed out vs. fell apart, overall."))

cells.append(nbf.v4.new_code_cell(
"""outcome_counts = df["outcome"].value_counts()
print(outcome_counts)

outcome_counts.plot(kind="bar", title="Outcome distribution (all 36 negotiations)")
plt.ylabel("count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell("## Outcome by language (EN vs KO)\n\nCore comparison for this study — n=18 per language here, far below what the full experiment will have."))

cells.append(nbf.v4.new_code_cell(
"""pd.crosstab(df["language"], df["outcome"])"""
))

cells.append(nbf.v4.new_markdown_cell("## Outcome by persona"))

cells.append(nbf.v4.new_code_cell(
"""pd.crosstab(df["persona"], df["outcome"])"""
))

cells.append(nbf.v4.new_markdown_cell("## Outcome by urgency"))

cells.append(nbf.v4.new_code_cell(
"""pd.crosstab(df["urgency"], df["outcome"])"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Final price, agreed negotiations only

Listing price for this scenario is $275, buyer target $137. Only rows with
`outcome == "agreed"` have a `final_price`."""
))

cells.append(nbf.v4.new_code_cell(
"""agreed = df[df["outcome"] == "agreed"]
print(f"{len(agreed)}/{len(df)} negotiations reached agreement")
agreed[["condition_id", "language", "urgency", "persona", "final_price", "n_turns"]]"""
))

cells.append(nbf.v4.new_code_cell(
"""if len(agreed) > 0:
    agreed.groupby("language")["final_price"].describe()"""
))

cells.append(nbf.v4.new_markdown_cell("## Turn count distribution\n\nHow many turns negotiations actually used (max is 10, per `negotiation.py`'s `MAX_TURNS`)."))

cells.append(nbf.v4.new_code_cell(
"""df["n_turns"].plot(kind="hist", bins=range(1, 12), title="Turns per negotiation")
plt.xlabel("n_turns")
plt.show()

df.groupby("outcome")["n_turns"].describe()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Next steps

- Schema and outcome/price extraction look correct (see above) — safe to run
  the full experiment (`python -m src.run_experiment --provider gemini
  --workers 8`, all 12 scenarios).
- No FTA/politeness-strategy coding yet — `judge.py` hasn't been run on
  these transcripts. That's a separate pass once the full transcript set
  exists.
- Re-run `build_smoke_test_analysis.py` against the full `results/transcripts/`
  once the 432-negotiation run finishes to regenerate this notebook with a
  real sample size."""
))

nb["cells"] = cells

client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
client.execute()

with open("notebooks/smoke_test_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote notebooks/smoke_test_analysis.ipynb")
