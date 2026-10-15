# Design changes log

Post-initial-design decisions worth carrying into the paper's Method or
Limitations sections. Distinct from `verification_log.md`, which is scoped
specifically to Korean translation-material verification — this file is for
changes to the experimental design itself, made after `research_statement.md`
was first written, along with the reasoning behind them.

## MAX_TURNS raised from 10 to 16 (2026-08-15)

**What changed:** `negotiation.py`'s `MAX_TURNS` (the per-negotiation turn
cap) went from 10 to 16.

**Why 10 was chosen originally:** per `research_statement.md`'s "Scenario
count and repetitions" section, 10 was picked mainly to keep total API calls
within Gemini's *free-tier* daily quota (~1,500 requests/day) across the
original execution window — not derived from negotiation-quality reasoning.
It was set close to CraigslistBargain's own reported average human dialogue
length (~9 turns) as a secondary justification.

**Why it was revised:** that free-tier constraint no longer applies —
negotiation calls now run through Vertex AI on a paid GCP project (see
`GEMINI_ADC_SETUP.md`), and the full 432-negotiation experiment's actual
cost is roughly $1-2 based on observed token usage, so call-count budget is
not a real constraint anymore.

A 36-negotiation smoke test (1 scenario, all 12 conditions x 3 reps) showed
23/36 hitting `max_turns_reached`. Inspecting the buyer/seller price gap at
the final turn showed two distinct patterns:
- **cooperative/neutral conditions:** small gaps (10-15 in several cases,
  e.g. `en-none-cooperative` rep0/rep1, `en-strong-cooperative` rep0/rep2,
  `ko-none-cooperative` rep0/rep2) — these read as negotiations that were
  still actively converging when the cap cut them off, i.e. an artifact of
  the cap rather than a real stall.
- **headstrong conditions:** large gaps (80-135, e.g. `en-none-headstrong`,
  `ko-none-headstrong`, `en-strong-headstrong`) — consistent with that
  persona's designed behavior ("concede slowly and reluctantly"), not
  obviously turn-limited. Raising the cap may not resolve these, and that's
  expected rather than a bug — a lower agreement rate for headstrong buyers
  could itself be a meaningful result, not just a measurement artifact.

**Decision:** raise the cap to 16 to let the near-converging cooperative/
neutral cases resolve, while accepting headstrong negotiations may still
often hit the (now higher) cap.

**Note for paper:** the turn cap was empirically revisited after a pilot
smoke test rather than fixed a priori purely from the CraigslistBargain
dataset average — worth stating in Method rather than presenting 16 as if
it were the original design choice. Also worth flagging as a limitation:
this smoke-test evidence comes from a single scenario (mattress/box-spring/
frame, listing $275) — the same MAX_TURNS value is applied uniformly
across all 12 scenarios without re-validating the near-miss pattern holds
for scenarios with very different price ranges (e.g. the $11,000 car or
$1,800/month housing listings), since a fixed absolute price gap (e.g. $15)
means something very different at those scales.

## Negotiation register (구어체 vs 문어체) — no change made (2026-08-15)

**Discussion:** generated negotiation utterances (both EN and KO) read as
complete, well-formed prose sentences rather than clipped, natural chat/text
messages — noticeably more "written" than how a real Danggeun Market or
Craigslist chat negotiation typically sounds.

**Why no change was made:** this pattern appears symmetrically in both
languages (the English transcripts are equally "prose-like," not just the
Korean ones), so it isn't a between-language confound for this study's core
comparison (does the *size* of the persona effect differ by language). The
study's dependent variables are price outcomes and FTA/politeness-strategy
coding, not chat-authenticity — and complete sentences arguably make
politeness-strategy markers (hedging, indirect requests) easier for the
judge model to detect than clipped chat shorthand would.

**Note for paper:** worth a one-line limitation that generated dialogue is
more formally "written" than authentic chat-negotiation text would be —
relevant if reviewers raise ecological validity, but doesn't threaten the
EN vs. KO comparison itself since the effect is symmetric across conditions.
