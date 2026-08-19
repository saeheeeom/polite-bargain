# Full Experiment Analysis Guide

This document explains the purpose, methods, outputs, results, and defensible
conclusions of [`notebooks/full_experiment_analysis.ipynb`](notebooks/full_experiment_analysis.ipynb).
It is written as both a reading guide for the notebook and a bridge from the
analysis to the paper.

The numerical results below are a snapshot of the completed analysis currently
stored under `results/analysis/`. The generated analysis directory is ignored by
Git, while this guide and the notebook remain available at the repository level.

## Executive Summary

- The experiment attempted **432 negotiations**: 12 scenarios x 2 languages x
  2 urgency conditions x 3 buyer personas x 3 repetitions.
- **430 negotiations were analyzed**. Two English, strong-urgency, headstrong
  runs ended in semantic `parse_error` outcomes and were excluded.
- The valid set contains **279 agreements**, **118 abandonments**, **33 turn-cap
  outcomes**, and **4,580 turns**.
- The scenario-balanced agreement rate was **52.5% in English** and **77.3% in
  Korean**. The Korean-minus-English effect was **+24.8 percentage points**, 95%
  CI [15.7, 34.7], exact p = .0010, Holm-adjusted p = .0039.
- The language effect depended strongly on persona. Korean had essentially no
  agreement advantage for cooperative buyers because both language conditions
  were already near ceiling. Its advantage was **+48.6 points for neutral** and
  **+25.7 points for headstrong** buyers.
- Strong seller urgency raised agreement from **50.0% to 79.9%** overall. The
  language-by-urgency estimate was positive, but its exact adjusted p-value was
  .113. It is suggestive, not a confirmed headline interaction.
- Among agreements, Korean produced a lower normalized final price than English
  by **0.250 bargaining-span units**, favoring the buyer. This is a conditional
  result and must not be described as the effect on all negotiations.
- Buyer opening anchors changed little by language. The clearest process result
  is that sellers conceded much more in Korean, buyers conceded somewhat less,
  and terminal buyer-seller gaps were smaller.
- FTA incidence was similar across languages. Korean turns were more often
  mitigated and less often bald-on-record, especially for sellers. These judge
  annotations are secondary descriptions, not evidence that politeness caused
  the objective outcome differences.
- The defensible scope is one LLM negotiation system under English versus Korean
  prompting. The experiment does **not** establish claims about human English or
  Korean negotiators, national cultures, or politeness as a causal mediator.

## 1. Research Goals

The notebook answers four paper-level questions:

1. Does English versus Korean prompting change agreement, final price, or
   negotiation length?
2. Does language change the effects of seller urgency and buyer persona?
3. Where do those differences appear in buyer anchors, role-specific
   concessions, offer trajectories, and terminal gaps?
4. Do FTA incidence and politeness realization differ by language, role, and
   experimental condition?

The evidentiary order matters:

1. **Primary evidence:** structured actions and objective outcomes.
2. **Mechanism description:** anchors, concessions, gaps, and trajectories.
3. **Secondary evidence:** FTA and politeness labels from the judge model.

The notebook deliberately does not run a formal mediation analysis. It can show
that politeness and outcomes differ together, but not that one caused the other.

## 2. Experimental Design And Analysis Sample

The design is a 2 x 2 x 3 factorial experiment:

| Factor | Levels |
|---|---|
| Language | English, Korean |
| Seller urgency | None, strong |
| Buyer persona | Cooperative, neutral, headstrong |

The experiment uses 12 item scenarios and three repetitions in each of the 12
factorial conditions:

`12 scenarios x 12 conditions x 3 repetitions = 432 attempted negotiations`

The notebook derives this expected count from the loaded scenario set, the
declared factor levels, and the configured repetition count. It also discovers
the valid and excluded counts from transcript outcomes rather than asserting
that they must be 430 and 2. The values below are the results for the current
dataset, not fixed assumptions in the analysis code.

The scenario, rather than each generated transcript, is the central blocking
unit for uncertainty. This prevents the analysis from pretending that 430
generations based on only 12 underlying scenarios are 430 fully independent
experimental units.

### Outcome Counts

| Quantity | Count |
|---|---:|
| Attempted negotiations | 432 |
| Valid negotiations analyzed | 430 |
| Semantic parse-error exclusions | 2 |
| Aligned judged transcripts | 430 |
| Valid turns | 4,580 |
| Agreements | 279 |
| Abandonments | 118 |
| Turn-cap outcomes | 33 |

The raw overall agreement proportion is `279 / 430 = 64.9%`. Marginal and
factorial estimates reported elsewhere are scenario- and condition-balanced,
so they should be preferred over unbalanced raw pooling.

### Exactly What Was Excluded

All 432 transcript files are syntactically valid JSON. The two failures happened
inside the negotiation system when it could not parse an agent response into the
required structured action. Their outer transcript files still saved correctly
and recorded `outcome: "parse_error"`.

| Excluded transcript | Usable turn records before error | Failure |
|---|---:|---|
| `scenario-00_en-strong-headstrong_rep1.json` | 5, followed by the error record | Malformed object-like response: expected a quoted property name |
| `scenario-06_en-strong-headstrong_rep0.json` | 14, followed by the error record | Plain prose was returned instead of a JSON action object |

The corresponding judged files are absent:

- `scenario-00_en-strong-headstrong_rep1_judged.json`
- `scenario-06_en-strong-headstrong_rep0_judged.json`

This is intentional. `src/judge.py` skips transcripts whose outcome is
`parse_error`. There are therefore **zero missing judge files among the 430
valid transcripts**. The visible `scenario-00` rep0 and rep2 judged files are
present; rep1 is the excluded run.

The affected `en-strong-headstrong` cell has 34 valid runs. Every other full
condition cell has 36. The notebook balances by scenario and condition so those
two exclusions do not give other cells extra weight.

## 3. Statistics In Simple Terms

### Normalized Price

The notebook puts every scenario on a common bargaining scale:

```text
(price - buyer_target) / (seller_target - buyer_target)
```

- `0` means the buyer's target.
- `1` means the seller's target/listing price.
- Lower agreed values favor the buyer.
- Higher agreed values favor the seller.
- A negative buyer opening means the buyer opened below their own target.
- Values are not clipped, although all observed agreed final prices passed the
  check that they lie between 0 and 1.

Example: if the buyer target is $800, the seller target is $1,000, and the
agreement is $900, normalized price is `(900 - 800) / (1000 - 800) = 0.5`.

### Scenario-Balanced Mean

The notebook first averages repetitions within each scenario-condition cell,
then averages the relevant cells while giving each scenario equal weight. In
simple terms, each product or situation gets one comparable vote. A scenario
with an extra valid repetition cannot dominate the estimate.

This is why the `en-strong-headstrong` scenario-balanced agreement estimate is
54.2%, even though the raw cell count is 18 agreements among 34 valid files.
The paper should report the estimate together with its run and scenario counts,
not reconstruct it as a raw fraction.

### Bootstrap Confidence Interval

The notebook resamples the 12 scenarios with replacement 10,000 times using
seed 42. It recalculates the statistic for each resample. The middle 95% of
those values is the 95% confidence interval.

Simple reading:

- Narrow intervals mean the scenario-level estimate is comparatively stable.
- Wide intervals mean scenarios vary or evidence is limited.
- An interval that crosses zero does not give clear directional evidence for a
  difference under this procedure.
- The interval reflects variation across these 12 scenario blocks. It does not
  by itself prove generalization to all negotiation domains or people.

### Factorial Contrast

A contrast is a planned difference between conditions:

- Language effect: Korean minus English.
- Urgency effect: strong urgency minus no urgency.
- Cooperative effect: cooperative minus neutral.
- Headstrong effect: headstrong minus neutral.

For agreement, a contrast of `+0.248` means an increase of 24.8 percentage
points. For normalized final price, a contrast of `-0.250` means a reduction of
one quarter of the buyer-to-seller target span.

### Interaction

An interaction is a difference between differences. For example, the
language-by-urgency interaction asks:

```text
(Korean strong - Korean none) - (English strong - English none)
```

A positive value means urgency had a larger effect in Korean. It does not mean
that every individual cell is high, nor does it identify why the difference
arose.

### Exact Scenario-Level Sign-Flip Test

For each of 12 scenarios, the notebook calculates a paired contrast. Under the
null hypothesis, its sign could be reversed without changing the distribution.
With 12 scenarios there are `2^12 = 4,096` possible sign assignments, so the
notebook can enumerate the exact reference distribution.

This test asks whether the effect is consistent across scenarios. It is more
appropriate than treating all 430 negotiations as independent observations.

### Holm Correction

Four agreement hypotheses receive exact p-values. Testing several hypotheses
raises the chance of at least one false positive. Holm correction adjusts those
p-values while being less conservative than a single Bonferroni adjustment.

The adjusted `holm_p` value is the one to use for confirmatory language in the
paper. The other factorial contrasts have bootstrap intervals but no formal
exact p-values in this notebook.

### Why A Bootstrap CI And Exact Test Can Disagree

The language-by-urgency agreement interaction has a bootstrap interval just
above zero, but exact p = .113. The two procedures ask related but nonidentical
questions. The bootstrap describes the distribution of the average estimate
under scenario resampling; the sign-flip test asks how consistently the paired
effect points in one direction across only 12 scenarios. For the paper, follow
the preregistered inferential rule: call this interaction suggestive, not
confirmed.

### Conditional Final Price

A final price exists only when an agreement occurs. Because language, urgency,
and persona also change whether agreement occurs, the set of negotiations
contributing prices differs across conditions.

Therefore, write:

> Among negotiations that reached agreement, Korean had lower normalized final
> prices than English.

Do not write:

> Korean lowered prices across all negotiations.

Only two scenarios had at least one agreement in every one of the 12 full
condition cells. The notebook still uses all available agreement data by cell,
but this sparse complete-cell coverage makes the selection caveat especially
important.

## 4. Metrics And What They Mean

### Primary Outcomes

| Metric | Definition | Interpretation |
|---|---|---|
| Agreement rate | Share ending in `agreed` | Higher means agreement was more likely |
| Abandonment rate | Share ending in `abandoned` | Higher means one agent explicitly walked away |
| Turn-cap rate | Share ending at 16 turns | Higher means more unresolved negotiations |
| Number of turns | Valid dialogue turns before termination | Measures interaction length, not efficiency by itself |
| Normalized final price | Accepted price on the 0-to-1 target scale | Lower favors buyer; defined only for agreements |

### Opening And Concession Metrics

Only `offer` and `counter-offer` actions create price points. An `accept` or
`reject` action does not invent a new offer.

| Metric | Definition | Interpretation |
|---|---|---|
| First offer | Role's first price-bearing action | Buyer anchor; seller opening is fixed at 1 by design |
| Offer count | Number of price-bearing actions by a role | How many price positions the role stated |
| Concession count | Moves toward the opponent | Buyer raises; seller lowers |
| Concession rate | Positive concessions divided by opportunities after the first offer | How often a role moved in the accommodating direction |
| Gross concession | Sum of all movements toward the opponent | Total accommodating movement, even if later reversed |
| Net concession | Directed difference from first to last role offer | Endpoint movement toward the opponent |
| Hardening | Sum of moves away from the opponent | Captures reversals or tougher subsequent positions |
| Mean concession size | Average positive concession | Typical size when the role actually conceded |
| First-concession turn | Transcript turn of the first positive concession | When movement began |
| Terminal gap | Last seller ask minus last buyer offer, normalized | Smaller means final positions were closer |

Positive net concession always means movement toward the opponent because the
sign is role-adjusted. For buyers, price increases count as concessions. For
sellers, price decreases count as concessions.

### Offer Trajectories

Negotiations have different numbers of offers. For each role, the notebook maps
the first price-bearing action to progress 0 and the last to progress 1, then
linearly interpolates 21 equally spaced points.

This makes average paths visually comparable. Progress is **role-specific offer
progress**, not literal clock time and not a set of 21 independent tests. The
trajectory figures are descriptive process visualizations.

### Pragmatic Metrics

The judge labels each turn for face-threatening acts (FTAs), mitigation, and a
politeness strategy. The notebook first calculates rates within each transcript
and role, then balances those transcript-level rates by scenario.

| Metric | Simple meaning |
|---|---|
| FTA rate | Share of role turns coded as making a potentially face-threatening move |
| Mitigation rate | Share of FTAs coded as softened or mitigated |
| Bald-on-record share | Direct, unsoftened FTA realization |
| Positive-politeness share | FTA realization using rapport or common ground |
| Negative-politeness share | FTA realization respecting autonomy or reducing imposition |
| Off-record share | Indirect or hinted FTA realization |

Strategy shares and mitigation rates are conditional on a turn being labeled an
FTA. These are judge-model measurements, not direct ground truth about culture.

## 5. Cell-By-Cell Notebook Walkthrough

### Cell 1: Research Contract

**What it does:** Defines the design, four research goals, primary versus
secondary evidence, normalization formula, exclusion rule, scenario balancing,
bootstrap, exact tests, and interpretation limits.

**Statistics:** None are run here. This is the analysis contract.

**Paper use:** Adapt it into the Research Questions, Design, and Statistical
Analysis portions of Methods.

### Cell 2: Imports And Configuration

**What it does:** Loads NumPy, pandas, and Matplotlib; discovers the repository
root; defines data/output paths; sets seed 42, 10,000 bootstrap samples, 95%
intervals, the 16-turn cap, factor levels, roles, actions, and plot settings.

**Statistics:** None yet. The fixed seed and paths make later outputs
reproducible.

**Paper use:** Usually no prose beyond reporting software, seed, number of
bootstrap draws, confidence level, and turn cap.

### Cell 3: Loading Explanation

**What it does:** Explains the difference between malformed JSON and a valid
transcript file whose recorded semantic outcome is `parse_error`.

**Statistics:** None.

**Paper use:** Supports the exclusion rule and transparent flow count.

### Cell 4: Load Design And Files

**What it does:** Loads the scenarios, validates required fields and target
spans, checks Korean fields, constructs the expected design keys from the
loaded scenarios, factor levels, and repetition count, and defines file hashing
helpers. For the current dataset this yields 432 expected transcript files.

**Statistics:** File and design counts only.

**Paper use:** Report 12 scenarios, 12 factorial cells, three repetitions, and
432 attempted negotiations.

### Cell 5: Validation Explanation

**What it does:** States the transcript, language, action, outcome, and judge
alignment checks that follow.

**Statistics:** None.

**Paper use:** Briefly state that automated integrity checks were applied before
analysis.

### Cell 6: Deep Validation And Data Loading

**What it does:** Parses filename metadata and validates condition fields,
repetitions, roles, alternation, allowed actions, prices, terminal actions,
outcomes, final-price consistency, basic English/Korean script expectations,
and one-to-one turn alignment with judged annotations. It discovers every
semantic `parse_error` and excludes those keys without assuming their number.

**Output:** 430 valid transcripts, 4,580 valid turns, and 430 aligned judged
files produced by `qwen/qwen3.7-flash`.

**Statistics:** Quality-control assertions, not hypothesis tests.

**Paper use:** Data quality and exclusion paragraph in Methods.

### Cell 7: QC Tables

**What it does:** Displays data-derived total counts, counts for all 12 design
cells, and the explicit exclusion table. Cell checks require attempted runs to
match the configured design and valid plus excluded runs to partition each
cell; they do not hard-code the currently deficient cell.

**Output:** 279 agreements, 118 abandonments, 33 turn-cap outcomes; 34 valid
runs in `en-strong-headstrong` and 36 in every other cell.

**Statistics:** Descriptive counts.

**Paper use:** Sample-flow statement and, if space permits, an appendix table.

### Cell 8: Metric Definitions

**What it does:** Defines which actions carry prices and how concession,
hardening, and endpoint movement are calculated for each role.

**Statistics:** None.

**Paper use:** Measures subsection.

### Cell 9: Construct Transcript And Pragmatic Frames

**What it does:** Converts each negotiation into one transcript-level row;
builds buyer, seller, and overall pragmatic rows; calculates outcome indicators,
terminal actor, openings, offer counts, concession measures, hardening, first
concession timing, final gap, and pragmatic rates.

**Checks:** One transcript row per valid key; buyer, seller, and overall
pragmatic rows per valid transcript; seller openings equal 1; and all agreed
prices lie within the target interval. This currently produces 430 transcript
rows and 1,290 pragmatic rows.

**Statistics:** Per-transcript metric construction, not group inference.

**Paper use:** Measures subsection and reproducibility appendix.

### Cell 10: Balancing Explanation

**What it does:** Explains why repetitions are averaged inside
scenario-condition cells before marginal comparisons.

**Statistics:** None.

**Paper use:** Statistical Analysis subsection.

### Cell 11: Primary Scenario-Balanced Summaries

**What it does:** Defines deterministic scenario bootstrap functions and
calculates agreement, abandonment, turn-cap rate, turns, and conditional final
price for every full cell and each factor marginal.

**Statistics:** Scenario-balanced means and 10,000-sample scenario-block
bootstrap 95% confidence intervals.

**Main output:** English agreement 52.5%, Korean 77.3%; no urgency 50.0%, strong
urgency 79.9%; cooperative 98.6%, neutral 49.3%, headstrong 46.9%.

**Paper use:** Main descriptive outcomes table.

### Cell 12: Contrast Orientation

**What it does:** Defines positive language effects as Korean minus English,
positive urgency effects as strong minus none, and persona contrasts relative
to neutral.

**Statistics:** None.

**Paper use:** Essential table-note language so signs cannot be misread.

### Cell 13: Factorial Contrasts And Headline Tests

**What it does:** Computes 77 contrasts across seven outcomes or behavioral
metrics and 11 main, two-way, or exploratory three-way comparisons. It runs four
exact scenario-level sign-flip tests for prespecified agreement contrasts and
applies Holm correction to those four p-values.

**Statistics:** Scenario-block bootstrap intervals, exact paired sign-flip
tests, and Holm multiple-testing correction.

**Main output:** Korean minus English agreement = +24.8 points, 95% CI [15.7,
34.7], Holm p = .0039.

**Paper use:** Main inferential table. Put the remaining contrasts in an
appendix and label three-way interactions exploratory.

### Cell 14: Behavioral Analysis Explanation

**What it does:** Clarifies interpretation of signed concession metrics and the
fixed seller opening.

**Statistics:** None.

**Paper use:** Short lead-in to the process/mechanism analysis.

### Cell 15: Openings, Concessions, And Terminal Gaps

**What it does:** Summarizes offer count, concession count/rate, gross and net
movement, hardening, mean concession size, first-concession timing, and final
offer gap. It creates language-by-urgency and language-by-persona views.

**Statistics:** Scenario-balanced means and scenario-bootstrap intervals.

**Main output:** Buyer opening differs little by language, while seller net
concession is 0.349 span units larger in Korean and terminal gaps are 0.286
units smaller.

**Paper use:** Central process table and mechanism figure.

### Cell 16: Trajectory Explanation

**What it does:** Defines the common role-specific progress scale used to
compare negotiations of unequal length.

**Statistics:** None.

**Paper use:** Figure caption or Measures detail.

### Cell 17: Offer Trajectories

**What it does:** Interpolates each buyer and seller offer sequence at 21
progress points, averages by scenario and condition, and creates full-cell,
language-by-urgency, and language-by-persona trajectory summaries.

**Statistics:** Scenario-balanced mean at each progress point with a
scenario-bootstrap interval.

**Main output:** Korean and urgent seller paths fall more sharply; Korean buyer
paths generally move less. This is consistent with the endpoint concession
metrics.

**Paper use:** Visual process evidence. Do not treat each plotted point as a
separate hypothesis test.

### Cell 18: Pragmatic Analysis Explanation

**What it does:** Establishes transcript-level rather than pooled-turn rates and
states that mitigation and strategy shares are conditional on FTAs.

**Statistics:** None.

**Paper use:** Pragmatics Measures subsection and interpretation caveat.

### Cell 19: FTA And Politeness Analysis

**What it does:** Summarizes FTA rate, mitigation among FTAs, and four strategy
shares by full condition, language and role, and language, role, and outcome.

**Statistics:** Scenario-balanced transcript-level means and bootstrap
intervals. No mediation test is run.

**Main output:** FTA frequency is similar; Korean uses more mitigation and
positive-politeness coding and less bald-on-record realization.

**Paper use:** Secondary Results subsection or appendix.

### Cell 20: Figure Explanation

**What it does:** States that all error bars and ribbons are 95% scenario-block
bootstrap intervals and that figures are saved as PNG and PDF.

**Statistics:** None.

**Paper use:** Shared figure-note language.

### Cell 21: Primary Outcome Figures

**What it does:** Produces interaction plots for agreement, conditional final
price, and negotiation length across language, urgency, and persona.

**Statistics shown:** Scenario-balanced point estimates and 95% intervals.

**Paper use:** In the final four-page manuscript, `agreement_interactions` is
reported in the appendix beside the per-scenario agreement table.
`final_price_interactions` needs a prominent conditional-on-agreement note.

### Cell 22: Mechanism Figures

**What it does:** Produces the buyer-opening and role-specific net-concession
plots.

**Main output:** Opening anchors are not the main language difference; seller
movement is.

**Paper use:** `net_concessions_language_urgency` is the clearest mechanism
figure. The opening plot is useful as supporting evidence against an anchor-only
explanation.

### Cell 23: Trajectory Figures

**What it does:** Plots buyer and seller offer paths for language by urgency and
language by persona.

**Main output:** Shows when and how the endpoint differences accumulated.

**Paper use:** One trajectory figure can appear in the main text; the other can
move to the appendix if space is tight.

### Cell 24: Pragmatic Figures

**What it does:** Produces language-by-role FTA/mitigation bars and politeness
strategy composition.

**Paper use:** Secondary result or appendix. These should follow, not replace,
the objective outcome and action analyses.

### Cell 25: Export Explanation

**What it does:** Explains that row-level exports omit raw utterance text and
that the generated Markdown summary is an audit aid rather than final paper
prose.

**Statistics:** None.

### Cell 26: Export And Manifest

**What it does:** Saves six derived data CSVs, 12 analysis-table CSVs, six LaTeX
tables, nine figures in both PNG and PDF, `quality_report.json`, and
`analysis_summary.md`. It records input hashes and software versions.

**Statistics:** No new analysis. This cell serializes prior results.

**Paper use:** Use the LaTeX tables and PDF figures as the starting point for the
manuscript, then edit captions and table scope to match the claims.

### Cell 27: Audit View Explanation

**What it does:** Introduces a compact final summary that can be inspected
without searching through earlier output.

**Statistics:** None.

### Cell 28: Audit Display

**What it does:** Displays `results/analysis/analysis_summary.md` inside the
notebook.

**Statistics:** None. It adds no new result.

**Current state:** All analytical and export code through Cell 26 has execution
counts and stored output. Cell 28 is currently unexecuted, which affects only
the convenience display, not any calculation or exported artifact.

## 6. Detailed Results

### 6.1 Full Experimental Cells

These are scenario-balanced cell estimates. Final price is conditional on
agreement.

| Language | Urgency | Persona | Valid | Agreements | Agreement | Mean turns | Final price |
|---|---|---|---:|---:|---:|---:|---:|
| English | None | Cooperative | 36 | 36 | 100.0% | 9.06 | 0.939 |
| English | None | Neutral | 36 | 4 | 11.1% | 8.61 | 0.270 |
| English | None | Headstrong | 36 | 5 | 13.9% | 13.89 | 0.705 |
| English | Strong | Cooperative | 36 | 35 | 97.2% | 8.56 | 0.751 |
| English | Strong | Neutral | 36 | 14 | 38.9% | 9.86 | 0.092 |
| English | Strong | Headstrong | 34 | 18 | 54.2% | 12.79 | 0.320 |
| Korean | None | Cooperative | 36 | 35 | 97.2% | 9.81 | 0.704 |
| Korean | None | Neutral | 36 | 18 | 50.0% | 12.28 | 0.251 |
| Korean | None | Headstrong | 36 | 10 | 27.8% | 13.19 | 0.122 |
| Korean | Strong | Cooperative | 36 | 36 | 100.0% | 8.03 | 0.428 |
| Korean | Strong | Neutral | 36 | 35 | 97.2% | 11.44 | 0.040 |
| Korean | Strong | Headstrong | 36 | 33 | 91.7% | 10.50 | 0.034 |

The English, no-urgency neutral and headstrong final-price cells contain only
four and five agreements. Those prices are especially selection-sensitive and
should not carry a standalone argument.

### 6.2 Marginal Primary Outcomes

| Factor level | Valid runs | Agreements | Agreement [95% CI] | Abandoned | Turn cap | Turns [95% CI] | Final price [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 214 | 112 | 52.5% [42.6, 64.1] | 39.1% | 8.3% | 10.46 [9.78, 11.08] | 0.513 [0.435, 0.581] |
| Korean | 216 | 167 | 77.3% [68.5, 86.1] | 15.7% | 6.9% | 10.88 [10.07, 11.61] | 0.263 [0.217, 0.318] |
| No urgency | 216 | 108 | 50.0% [39.8, 62.0] | 39.8% | 10.2% | 11.14 [10.64, 11.57] | 0.499 [0.391, 0.585] |
| Strong urgency | 214 | 171 | 79.9% [71.3, 88.0] | 15.0% | 5.1% | 10.20 [9.25, 11.02] | 0.278 [0.248, 0.312] |
| Cooperative | 144 | 142 | 98.6% [95.8, 100.0] | 0.7% | 0.7% | 8.86 [8.00, 9.78] | 0.706 [0.647, 0.759] |
| Neutral | 144 | 71 | 49.3% [36.8, 63.2] | 50.0% | 0.7% | 10.55 [9.88, 11.10] | 0.163 [0.129, 0.195] |
| Headstrong | 142 | 66 | 46.9% [35.1, 60.1] | 31.6% | 21.5% | 12.59 [11.75, 13.38] | 0.295 [0.191, 0.384] |

What this says:

- Korean primarily converts abandonments into agreements. Turn-cap rates differ
  much less by language.
- Strong urgency is associated with much more agreement, fewer abandonments,
  fewer caps, and slightly shorter negotiations.
- Cooperative buyers nearly always agree, but conditional prices are much more
  seller-favorable. Their high agreement rate comes with economic accommodation.
- Neutral buyers either obtain low prices or abandon, producing a 50%
  abandonment rate.
- Headstrong buyers do not improve agreement relative to neutral buyers overall
  and are far more likely to hit the turn cap. Rigidity lengthens unresolved
  bargaining.

### 6.3 Confirmatory Agreement Contrasts

| Contrast | Estimate | 95% CI | Exact p | Holm p | Conclusion |
|---|---:|---:|---:|---:|---|
| Korean - English | +0.248 | [0.157, 0.347] | .0010 | .0039 | Clear language-condition effect |
| Language x urgency | +0.162 | [0.009, 0.329] | .1133 | .1133 | Suggestive, not confirmed |
| Language x (cooperative - neutral) | -0.486 | [-0.667, -0.306] | .0010 | .0039 | Language effect is much smaller for cooperative than neutral |
| Language x (headstrong - neutral) | -0.229 | [-0.368, -0.083] | .0195 | .0391 | Language effect is smaller for headstrong than neutral |

The language main effect and both persona interactions survive Holm correction.
The language-by-urgency interaction does not.

### 6.4 Agreement Simple Effects

#### Language By Urgency

| Language | No urgency | Strong urgency | Urgency increase |
|---|---:|---:|---:|
| English | 41.7% | 63.4% | +21.8 points |
| Korean | 58.3% | 96.3% | +38.0 points |

Urgency appears more powerful in Korean, producing the +16.2-point interaction.
Because exact Holm p = .113, present this as a visible pattern that needs more
scenario blocks, not as a settled interaction.

#### Language By Persona

| Persona | English | Korean | Korean - English |
|---|---:|---:|---:|
| Cooperative | 98.6% | 98.6% | 0.0 points |
| Neutral | 25.0% | 73.6% | +48.6 points |
| Headstrong | 34.0% | 59.7% | +25.7 points |

This is the most important interaction result. The overall language effect is
not uniform. Cooperative prompting nearly guarantees agreement in either
language, leaving no room for a Korean advantage. The language difference
appears mainly when the buyer policy permits meaningful disagreement.

#### Urgency By Persona

- Cooperative agreement is 98.6% with or without urgency, again showing a
  ceiling.
- Neutral agreement rises from 30.6% to 68.1% under strong urgency.
- Headstrong agreement rises from 20.8% to 72.9% under strong urgency.

Urgency therefore matters most when the buyer is not already cooperative.

### 6.5 Conditional Final Price

Main contrasts among the 279 agreements:

| Contrast | Estimate | 95% CI | Meaning |
|---|---:|---:|---|
| Korean - English | -0.250 | [-0.315, -0.178] | Korean agreements end closer to the buyer target |
| Strong urgency - none | -0.221 | [-0.318, -0.107] | Urgent-seller agreements end closer to the buyer target |
| Cooperative - neutral | +0.542 | [0.488, 0.591] | Cooperative buyers accept much more seller-favorable prices |
| Headstrong - neutral | +0.132 | [0.035, 0.202] | Headstrong agreements are more seller-favorable than neutral agreements |
| Language x urgency | +0.059 | [-0.062, 0.140] | No clear conditional-price interaction |
| Language x (cooperative - neutral) | -0.243 | [-0.338, -0.144] | Persona differences in agreed price vary by language |
| Language x (headstrong - neutral) | -0.399 | [-0.546, -0.207] | Headstrong-versus-neutral price differences vary strongly by language |

The likely substantive reading is that Korean and strong urgency make the
seller policy more willing to move, while cooperative buyers secure agreement
by conceding more themselves. However, agreement is a post-treatment filter.
These prices cannot establish what every attempted negotiation would have paid.

### 6.6 Negotiation Length And Terminal Actors

Key turn-count contrasts:

- Korean minus English: +0.41 turns, 95% CI [-0.17, 1.07]. There is no clear
  overall language difference in length.
- Strong urgency minus none: -0.94 turns, 95% CI [-1.58, -0.33].
- Cooperative minus neutral: -1.69 turns, 95% CI [-2.36, -0.99].
- Headstrong minus neutral: +2.05 turns, 95% CI [1.36, 2.81].
- Language by urgency: -1.65 turns, 95% CI [-2.81, -0.57].
- Language by cooperative-versus-neutral: -2.51 turns, 95% CI [-3.88, -1.21].
- Language by headstrong-versus-neutral: -4.12 turns, 95% CI [-4.93, -3.36].

These are bootstrap contrasts, not part of the four exact p-value tests. They
show that negotiation length is shaped more by urgency and persona interactions
than by a simple language main effect.

Terminal actions provide a useful audit:

- Buyers accepted in 169 agreements; sellers accepted in 110.
- Buyers walked away in 64 abandonments; sellers walked away in 54.
- All 33 capped transcripts end on a buyer turn because the fixed 16-turn
  alternating sequence ends with the buyer. That count should not be interpreted
  as buyers causing every turn cap.

### 6.7 Opening Anchors, Concessions, And Terminal Gaps

Main language contrasts:

| Metric | Korean - English | 95% CI | Interpretation |
|---|---:|---:|---|
| Buyer first offer | +0.024 | [-0.011, 0.056] | Little evidence of a general opening-anchor shift |
| Buyer net concession | -0.086 | [-0.137, -0.041] | Korean buyers move less toward sellers |
| Seller net concession | +0.349 | [0.237, 0.462] | Korean sellers move substantially more toward buyers |
| Terminal gap | -0.286 | [-0.400, -0.175] | Final buyer-seller positions are closer in Korean |

Language-by-urgency process means:

| Metric | EN none | EN strong | KO none | KO strong |
|---|---:|---:|---:|---:|
| Buyer first offer | -0.157 | -0.162 | -0.146 | -0.126 |
| Buyer net concession | 0.418 | 0.385 | 0.375 | 0.254 |
| Seller net concession | 0.075 | 0.360 | 0.401 | 0.732 |
| Buyer concession rate | 0.894 | 0.866 | 0.675 | 0.601 |
| Seller concession rate | 0.163 | 0.551 | 0.592 | 0.973 |
| Terminal gap | 0.665 | 0.418 | 0.370 | 0.139 |

Across urgency conditions, the language-level descriptive means reinforce the
same result:

- English buyer concession rate: 0.880; Korean: 0.638.
- English seller concession rate: 0.357; Korean: 0.783.
- English buyer gross concession: 0.404; Korean: 0.320.
- English seller gross concession: 0.225; Korean: 0.573.
- Hardening is small in both languages, so these endpoint patterns are not
  mainly caused by frequent reversals.

The sharpest cell is Korean with strong urgency: sellers concede on almost
every opportunity, move downward by 0.732 span units from first to last offer,
and finish only 0.139 span units from the buyer's last offer on average.

The clean mechanism conclusion is not that Korean buyers begin with radically
different anchors. It is that the generated seller policy responds differently:
more seller movement, less required buyer movement, and smaller terminal gaps.

### 6.8 Persona And Negotiation Policy

Buyer openings by persona and language:

- Cooperative: English -0.110; Korean -0.051.
- Neutral: English -0.366; Korean -0.294.
- Headstrong: English -0.003; Korean -0.064.

Neutral buyers actually open farthest below their target. Headstrong behavior
is therefore better characterized by later rigidity than by the most aggressive
opening anchor.

Buyer net concessions:

- English: cooperative 0.536, neutral 0.369, headstrong 0.299.
- Korean: cooperative 0.464, neutral 0.342, headstrong 0.139.

Seller net concessions:

- English: cooperative 0.149, neutral 0.257, headstrong 0.246.
- Korean: cooperative 0.416, neutral 0.688, headstrong 0.597.

These values explain the persona outcomes:

- Cooperative buyers reach agreement by moving substantially toward sellers.
- Neutral buyers make low openings and often abandon, but their successful deals
  are buyer-favorable.
- Headstrong buyers move the least, especially in Korean. They need large seller
  movement to agree and otherwise tend to prolong negotiations or hit the cap.

### 6.9 Offer Trajectories

The trajectory figures support, rather than replace, the endpoint metrics:

- Seller paths descend much more in Korean than in English.
- Strong urgency steepens seller descent in both languages.
- Korean buyer paths generally rise less, especially under strong urgency and
  headstrong persona.
- Persona differences emerge over the bargaining path, not only at the first
  offer.

Use the trajectories to explain *how* final prices and agreements emerged. Do
not claim statistical significance at individual progress points without a
separate simultaneous-inference procedure.

### 6.10 FTA And Politeness Results

| Role | Language | FTA rate | Mitigation among FTAs | Bald | Positive | Negative | Off-record |
|---|---|---:|---:|---:|---:|---:|---:|
| Buyer | English | 0.904 | 0.449 | 0.551 | 0.053 | 0.396 | 0.000 |
| Buyer | Korean | 0.916 | 0.516 | 0.484 | 0.146 | 0.369 | 0.002 |
| Seller | English | 0.868 | 0.036 | 0.964 | 0.007 | 0.030 | 0.000 |
| Seller | Korean | 0.899 | 0.117 | 0.883 | 0.045 | 0.071 | 0.000 |

Simple interpretation:

- Language does not appear to remove the underlying face-threatening nature of
  bargaining acts. FTA rates remain high in both languages.
- Korean realization is somewhat more softened: mitigation and positive
  politeness are higher, while bald-on-record shares are lower.
- The seller difference is proportionally large because English seller
  mitigation is extremely rare, but the Korean absolute seller mitigation rate
  is still only 11.7%.
- Off-record strategies are effectively absent, suggesting this model/judge
  pair represents bargaining mainly through direct strategies.

These patterns are compatible with the idea that linguistic realization and
negotiation policy shift together. They do not prove that mitigation produces
agreement. A causal mediation claim would require a separate design and model.

## 7. Overall Conclusions

### Strongly Supported Within This Experiment

1. **Prompt language changes this LLM negotiation system's behavior.** Korean
   prompting raises scenario-balanced agreement by 24.8 points, with a
   Holm-adjusted exact p-value of .0039.
2. **The language effect is context-dependent.** It is absent under cooperative
   buyer prompting, largest under neutral prompting, and intermediate under
   headstrong prompting. Both planned persona interactions survive Holm
   correction.
3. **Seller movement is the clearest behavioral location of the language
   difference.** Buyer openings barely shift overall, while Korean sellers make
   much larger concessions and finish with smaller gaps.
4. **Strong urgency is a powerful policy manipulation in this setup.** It is
   associated with more agreement, lower conditional prices, larger seller
   concessions, fewer caps, and shorter negotiations.
5. **Persona controls the agreement-price tradeoff.** Cooperative buyers agree
   almost universally but at seller-favorable prices. Neutral and headstrong
   buyers secure lower prices when they agree but fail more often.

### Supported As Secondary Description

1. Korean buyer and seller turns contain more mitigation and positive-politeness
   coding and less bald-on-record realization.
2. FTA incidence itself changes little, so the distinction is more about how
   threats to face are expressed than whether bargaining contains them.
3. Objective policy shifts and pragmatic shifts co-occur.

### Not Established

1. Politeness caused higher agreement or lower price.
2. Korean language generally improves negotiation for other models, prompts,
   tasks, populations, or real humans.
3. The agreed-price contrast represents all attempted negotiations.
4. The language-by-urgency interaction is confirmatory; adjusted exact p = .113.
5. Exploratory three-way interactions are reliable. Both agreement three-way
   intervals include zero.
6. Every pointwise difference in a trajectory figure is independently
   significant.

### Recommended One-Sentence Conclusion

> Under otherwise fixed conditions in this LLM negotiation system, Korean
> prompting increased agreement and produced more buyer-favorable prices among
> successful negotiations, primarily alongside greater seller concession; the
> agreement effect was concentrated in neutral and headstrong buyer policies
> rather than the near-ceiling cooperative condition.

## 8. What Should Go In The Paper

### Abstract

Include:

- `pilot-scale`.
- 430 analyzed negotiations. Keep the two technical failures out of the
  abstract and document 432 attempts, the exclusion rule, and both filenames in
  Methods and the appendix.
- The 2 x 2 x 3 design over 12 scenarios.
- The +24.8-point Korean-language minus English-language agreement contrast
  with adjusted p = .0039.
- The conditional agreed-price direction.
- The seller-concession mechanism.
- A restrained statement that language interacts with persona.

Keep the abstract focused. Pragmatic percentages and the unconfirmed
language-by-urgency interaction can remain outside it.

### Methods

Include:

- Models and exact prompting roles.
- Language, urgency, and persona manipulations.
- 12 scenarios and three repetitions.
- 432 attempted and 430 analyzed.
- Explicit semantic `parse_error` exclusion rule.
- Price normalization formula and interpretation.
- Definitions of agreement, abandonment, turn cap, and concession.
- Scenario balancing and scenario-block bootstrap with 10,000 draws, seed 42.
- Four exact scenario-level sign-flip tests and Holm correction.
- Final price conditional on agreement.
- Pragmatics as secondary, transcript-level analysis with no mediation.
- Translation procedure from [`verification_log.md`](verification_log.md):
  machine translation followed by a single fluent-speaker review, including
  removal of unsourced Korean-only additions and deliberate non-localization of
  US settings and USD prices.
- For annotation quality, say that a subset was manually spot-checked or
  calibrated. Avoid saying the procedure was formally validated unless the
  subset, criteria, disagreements, and resolution protocol are recorded.

### Results Order

1. Agreement contrast and four corrected headline tests.
2. Persona simple effects that explain the interactions.
3. Opening, concession, terminal-gap, and trajectory evidence.
4. Negotiation length as a supporting result.
5. Conditional final price with a selection warning.
6. Pragmatic realization as secondary description.

The data flow and two technical exclusions belong in Methods and the
reproducibility appendix rather than leading the Results section.

### Main Figures

Final four-page placement:

1. `offer_trajectories_language_persona.pdf` is the main-body figure because it
   shows how the agreement differences arise after similar buyer openings.
2. `agreement_interactions.pdf` appears in the appendix with the per-scenario
   agreement table.
3. `pragmatic_rates_language_role.pdf` and
   `politeness_strategy_composition.pdf` appear in the appendix, while their
   central descriptive results remain in the main Results section.

The conditional price interaction, net-concession figure, buyer-opening figure,
and turn-length figure remain notebook artifacts available for supplementary
analysis. The main text reports their key numerical contrasts directly.

### Main Tables

- Main Table 1 reports headline agreement contrasts, including corrected
  exact-test p-values.
- Appendix Table 2 reports agreement counts and rates separately for all 12
  scenarios and both interaction languages.
- Appendix Table 3 reports full-cell agreement, abandonment, turn-cap, and
  conditional-price outcomes.
- Appendix Tables 4 and 5 report judge-derived FTA, mitigation, and politeness
  strategy summaries by language and role.

### Limitations

State all of the following:

- This is a pilot-scale study with only 12 scenario blocks.
- Two failures were excluded from one English condition.
- Final price is conditional on agreement and vulnerable to selection.
- Only one negotiator model configuration and one judge model were studied.
- Pragmatic labels are model judgments, even with manual spot-checking.
- Translation verification used one fluent reviewer and was intentionally
  lightweight rather than a multi-annotator protocol.
- Korean scenarios are translations of US-located, USD-priced listings rather
  than fully localized Korean marketplace scenarios.
- Three-way interactions are exploratory.
- Simulated LLM negotiation behavior does not directly establish human cultural
  behavior.

## 9. Suggested Results Prose

The following is a safe starting point, not a substitute for fitting the prose
to the final paper style:

> Across 432 attempted negotiations, 430 completed runs were analyzed after two
> semantic response-parsing failures. Scenario-balanced agreement was 52.5% in
> English and 77.3% in Korean, a difference of 24.8 percentage points (95% CI
> [15.7, 34.7]; exact p = .0010; Holm-adjusted p = .0039). This language effect
> was not uniform across buyer policies: it was zero in the near-ceiling
> cooperative condition, 48.6 points under the neutral policy, and 25.7 points
> under the headstrong policy. The language-by-cooperative-versus-neutral and
> language-by-headstrong-versus-neutral contrasts remained significant after
> Holm correction.

> Process measures locate the principal difference in seller behavior rather
> than buyer anchoring. The Korean-minus-English shift in first buyer offer was
> small and its interval crossed zero (+0.024 span units, 95% CI [-0.011,
> 0.056]), whereas seller net concession was 0.349 units larger (95% CI [0.237,
> 0.462]) and the terminal offer gap was 0.286 units smaller (95% CI [-0.400,
> -0.175]). Among agreements, normalized final prices were 0.250 units lower in
> Korean (95% CI [-0.315, -0.178]); because agreement rates differed sharply,
> this price comparison is conditional and does not represent all attempted
> negotiations.

> Pragmatic annotation showed similar FTA incidence across languages but higher
> mitigation and positive-politeness realization in Korean. These results are
> descriptive: the experiment does not identify pragmatic realization as a
> mediator of agreement or price.

## 10. Generated Artifact Guide

The notebook currently produces **44 files** under `results/analysis/`.

### Audit Files

| File | Use |
|---|---|
| `analysis_summary.md` | Compact headline counts, tests, and cautions |
| `quality_report.json` | Analysis contract, exclusions, hashes, counts, and software versions |

### Derived Data

| File | Use |
|---|---|
| `data/attempt_manifest.csv` | All expected and observed attempts |
| `data/exclusions.csv` | Exact excluded runs and error messages |
| `data/transcript_metrics.csv` | One row per valid negotiation |
| `data/turn_metrics.csv` | Structured per-turn actions, prices, and judge labels without raw utterances |
| `data/pragmatic_transcript_metrics.csv` | Buyer, seller, and overall transcript-level pragmatic rates |
| `data/trajectory_points.csv` | Interpolated role-specific offer paths |

### Tables

| File | Use |
|---|---|
| `tables/design_cell_counts.csv` | Attempted, valid, and excluded counts by cell |
| `tables/primary_outcomes_by_cell.csv` | Full 12-cell outcome estimates |
| `tables/primary_outcomes_marginal.csv` | Language, urgency, and persona summaries |
| `tables/factorial_contrasts.csv` | All 77 main and interaction contrasts |
| `tables/headline_agreement_tests.csv` | Four exact agreement tests and Holm p-values |
| `tables/behavior_by_cell.csv` | All behavioral measures by full cell |
| `tables/behavior_by_language_urgency.csv` | Key process measures by language and urgency |
| `tables/behavior_by_language_persona.csv` | Key process measures by language and persona |
| `tables/trajectory_summary.csv` | Scenario-balanced trajectory estimates and intervals |
| `tables/pragmatics_by_cell_role.csv` | Pragmatic measures by full cell and role |
| `tables/pragmatics_by_language_role.csv` | Main descriptive pragmatic table |
| `tables/pragmatics_by_outcome_language_role.csv` | Pragmatics split by outcome, language, and role |

The tables directory also contains six LaTeX exports for direct manuscript use.

### Figures

Each figure is saved as PNG and PDF:

- `agreement_interactions`
- `final_price_interactions`
- `turn_length_interactions`
- `buyer_opening_anchors`
- `net_concessions_language_urgency`
- `offer_trajectories_language_urgency`
- `offer_trajectories_language_persona`
- `pragmatic_rates_language_role`
- `politeness_strategy_composition`

## 11. Reproducibility And Rerun Checklist

1. Run the notebook from the repository root with a clean kernel.
2. Confirm the expected grid is derived as 432 keys and all transcript keys
   match that grid.
3. Inspect the generated exclusion table; for the current dataset it contains
   exactly two semantic failures and leaves 430 valid transcripts.
4. Confirm valid plus excluded runs equal attempted runs in every cell; the
   current `en-strong-headstrong` cell has 34 valid runs and all others have 36.
5. Confirm 4,580 validated turns and one-to-one judge alignment for all 430
   valid transcripts.
6. Confirm all seller openings normalize to 1 and all agreed final prices fall
   between 0 and 1.
7. Confirm all 44 generated artifacts are recreated.
8. Re-run with seed 42 and compare deterministic tables, intervals, and input
   hashes in `quality_report.json`.

During the current verification pass on 2026-08-18, all 862 transcript and
judged JSON files decoded successfully. The notebook executed twice from clean
kernels with seed 42. All 26 CSV, LaTeX, JSON, and Markdown artifacts had
identical individual hashes across runs. Hashing their sorted relative paths
and contents produced this aggregate SHA-256:

```text
f12b7058b0f59638553eb2b1a81d01b790b697f5df0632132bc378cb6b0e4fd9
```

Because `results/analysis/` is intentionally ignored by Git, rerun the notebook
after cloning the repository or after changing any scenario, transcript, judged
annotation, or analysis code.

## 12. Bottom Line

The paper's strongest story is specific and internally coherent: in this fixed
LLM bargaining setup, Korean prompting raises agreement, especially for neutral
and headstrong buyer policies; buyer openings remain broadly similar, while
sellers concede much more and terminal gaps shrink. Conditional agreed prices
therefore favor buyers more in Korean. Pragmatic annotations show a parallel
shift toward more mitigated realization, but they should remain secondary
because the experiment does not identify politeness as the causal mechanism.
