"""
Codes a completed negotiation transcript for Face-Threatening Acts (FTA) and
politeness-mitigation strategy, per Brown & Levinson (1987).

Runs ONCE per full transcript (not per turn) — the judge sees the whole
negotiation and labels every turn in a single call, to keep coding cost low.

Per the research statement, use a DIFFERENT model as judge than whichever
model(s) generated the negotiation, to avoid a model evaluating its own output.
"""
import json
import re

from src.agents import call_agent

JUDGE_INSTRUCTIONS = """
You are a discourse analyst coding a negotiation transcript using Brown & Levinson's
politeness theory. For EACH turn in the transcript below, determine:

1. is_fta: true if the turn is a Face-Threatening Act (a request, refusal, or other
   utterance that puts social pressure on the other party), false otherwise.
2. strategy: if is_fta is true, classify the mitigation strategy as exactly one of:
   - "bald_on_record": direct, no cushioning
   - "positive_politeness": friendly framing, emphasizes closeness/goodwill
   - "negative_politeness": cautious, deferential, apologetic, hedged
   - "off_record": indirect, just a hint, not a direct ask
   If is_fta is false, set strategy to null.

Respond with ONLY a JSON array, one object per turn, in this shape:
[
    {{"turn": 0, "role": "seller", "is_fta": true, "strategy": "bald_on_record"}},
    {{"turn": 1, "role": "buyer", "is_fta": false, "strategy": null}},
  ...
]

Transcript:
{transcript}
"""

JUDGE_MODEL = "google/gemma-4-26b-a4b-it:free"  # different provider from the default Gemini negotiators


def format_transcript(turns: list[dict]) -> str:
    lines = []
    for i, t in enumerate(turns):
        lines.append(f"[{i}] {t['role']}: {t['utterance']}")
    return "\n".join(lines)


def code_transcript(turns: list[dict], judge_model: str = JUDGE_MODEL) -> list[dict]:
    prompt = JUDGE_INSTRUCTIONS.format(transcript=format_transcript(turns))
    raw = call_agent(judge_model, system_prompt=prompt, history=[], self_role="judge")
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in judge response: {raw!r}")
    coded = json.loads(match.group(0))
    if len(coded) != len(turns):
        print(f"WARNING: judge coded {len(coded)} turns but transcript has {len(turns)} — "
              f"check for parsing issues before trusting this result")
    return coded
