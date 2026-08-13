# Korean Translation Verification Log

Records the lightweight spot-check pass over machine-translated Korean materials,
per the verification plan in `research_statement.md` ("Translation and
verification (lightweight, by design)" and the "Translation verification is
lightweight, not exhaustive" limitation). Intended as source material for the
paper's Method and Limitations sections — not itself part of the paper.

- **Date:** 2026-08-12
- **Annotator:** single reviewer (project author), fluent Korean speaker — see
  "Single-annotator verification" limitation in research_statement.md
- **Scope reviewed:** all 12 scenario translations (`data/scenarios.json`,
  `title_ko`/`description_ko`) and all 5 urgency/persona phrasing strings
  (`src/translations.py`)
- **Translation source:** machine-translated via Gemini (`gemini-3.6-flash`),
  one read-through verification pass, no second annotator

## Findings

### 1. Systematic MT addition of unsourced CTA phrasing (corrected)

4 of 12 scenario descriptions had promotional closing lines added by the
translation model that had no corresponding text in the English source (e.g.
"편하게 채팅 주세용!", "궁금하신 점 편하게 문의 주세요!"). The model appears to
have inferred these as idiomatic for Korean secondhand-marketplace listings
(당근마켓 style), which the translation prompt explicitly asked for — but
because they exist only in the Korean condition, they risk being a
between-language confound: the Korean scenario text would read as
systematically warmer/more inviting than its English counterpart independent
of the persona/urgency manipulation being tested.

**Action:** removed the unsourced additions to restore 1:1 semantic
correspondence between `description` and `description_ko`. Affected items:
LG V10 phone, Raleigh bike, townhome listing, Waterstone listing.

**Note for paper:** worth naming explicitly as a translation-fidelity risk of
prompting for "natural" register — a literal-fidelity prompt would not have
introduced this, at some cost to naturalness. Also worth noting that this
mechanism (LLM adding culturally-idiomatic content beyond the source) is
distinct from mistranslation and might not be caught by a fluency-only review;
it required checking against the English source line by line.

### 2. Sentence-final register lacked natural variation (adjusted)

All 12 descriptions initially used uniform sentence-final polite style
(해요체/합니다체 with exclamation points). A Korean-speaker read of real
marketplace listings suggests noun-ending (명사형 종결 / 개조식) style is at
least as common, especially in spec-list-like descriptions.

**Action:** to avoid the opposite bias — a uniformly chatty register across
all 12 items that wouldn't reflect real listing variation — 4 of 12
descriptions were selected at random (`random.seed(42)`, `random.sample`) and
rewritten in noun-ending style: furniture-mattress, car-chevy-van,
bike-raleigh, furniture-china-hutch. The other 8 retain sentence-final style.

**Note for paper:** this is a deliberate register-diversity decision, not a
correction of an error — flag if a reviewer asks why register isn't uniform
across the corpus.

### 3. US addresses in Korean-language descriptions (considered, not changed)

Two scenarios (DJ gear listing, Waterstone housing listing) retain literal US
street addresses (e.g. "East Richmond Heights", "39600 FREMONT BLVD, Fremont,
CA 94538") inside the Korean-language description. Considered localizing these
to Korean addresses for naturalness, but decided against it:

- Listing prices remain in USD (`$`) in both language conditions
  (`negotiation.py`'s `BASE_PROMPT` embeds `${price}`/`${target}` regardless of
  language) — a Korean address paired with USD pricing would be a more
  internally inconsistent scenario than an English-language address kept
  as-is.
- Changing the geographic/cultural setting is adaptation, not translation — it
  would introduce a content difference between EN and KO conditions beyond
  the language manipulation itself, which is exactly the kind of confound the
  CTA-phrasing fix (above) was trying to remove in the other direction.
  Keeping foreign proper nouns untranslated is also standard practice in real
  Korean translations of foreign listings.
- Full localization (address + KRW conversion) would be a larger scope change
  cascading into `buyer_target`/`seller_target` and outcome comparability
  across languages — not attempted given the Aug 25, 2026 deadline.

**Note for paper:** state as a deliberate scope decision in Limitations,
alongside the existing lightweight-verification limitation — the Korean
condition describes USD-priced, US-located items translated into Korean,
not a fully localized Korean marketplace scenario.

### 4. Urgency and persona phrasing (5 strings, confirmed clean)

All 5 strings in `src/translations.py` (1 urgency + 2×2 persona variants,
"neutral" is empty by design) were read against their English source and
judged accurate and natural, in register consistent with each other
(formal 합니다체). No changes made. Also checked `negotiation.py`'s
`BASE_PROMPT` ko strings (buyer/seller framing sentence) — same result, no
changes made.

## Resulting state

All `# TODO: verify Korean` markers removed from `src/translations.py` and
`src/negotiation.py` — the lightweight verification pass described in
research_statement.md is complete for the current scenario set and phrasing
strings. If scenarios are resampled or phrasings edited later, this log
should be updated or a new pass logged.
