"""
Codes a completed negotiation transcript for Face-Threatening Acts (FTA) and
politeness-mitigation strategy, per Brown & Levinson (1987).

Runs ONCE per full transcript (not per turn) — the judge sees the whole
negotiation and labels every turn in a single call, to keep coding cost low.

Per the research statement, use a DIFFERENT model as judge than whichever
model(s) generated the negotiation, to avoid a model evaluating its own output.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.agents import call_agent

DEFAULT_WORKERS = 8

JUDGE_INSTRUCTIONS = """
You are a discourse analyst coding a negotiation transcript using Brown & Levinson's
politeness theory. For EACH turn in the transcript below, determine:

1. is_fta: true if the turn is a Face-Threatening Act (a request, refusal, or other
   utterance that puts social pressure on the other party), false otherwise.
2. strategy: if is_fta is true, classify the mitigation strategy as exactly one of:
   - "bald_on_record": the core request/refusal/assertion ITSELF is stated flatly,
     with no hedging on the substance — even if the turn also contains a polite
     opener or acknowledgment elsewhere. Example: "I appreciate the offer, but I'm
     firm at $275, that's my bottom dollar." is bald_on_record: the price stance is
     asserted with no softening ("maybe", "I wonder if", a question form) despite
     the polite-sounding opener.
   - "positive_politeness": friendly framing that appeals to shared interest or
     closeness ("let's both walk away happy here", "I want this to work for you too").
   - "negative_politeness": the core request/refusal ITSELF is hedged, indirect, or
     deferential — e.g. "Would you possibly be willing to consider $150?" or "I don't
     suppose you could go a little lower?". The hedge must be on the substance of the
     ask, not just a polite opener like "thanks for reaching out."
   - "off_record": indirect, just a hint, not a direct ask at all.

   Judge the directness of the CORE request or assertion, not just whether the turn
   contains any polite-sounding words. Do not default to negative_politeness just
   because a turn's overall tone is courteous — a flatly-stated refusal or firm price
   dressed in a polite opener is still bald_on_record.

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

JUDGE_MODEL = "qwen/qwen3.7-flash"  # different provider/family from the Gemini negotiators; paid tier
# to avoid the OpenRouter free-model daily-request cap (see design_changes_log.md)
JUDGED_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "judged"


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


def _load_transcript(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _default_output_path(input_path: Path) -> Path:
    return JUDGED_RESULTS_DIR / f"{input_path.stem}_judged{input_path.suffix}"


def _write_judged_transcript(input_path: Path, transcript: dict, coded_turns: list[dict], judge_model: str) -> Path:
    output_path = _default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_transcript": str(input_path),
        "judge_model": judge_model,
        "scenario_id": transcript.get("scenario_id"),
        "condition_id": transcript.get("condition_id"),
        "repetition": transcript.get("repetition"),
        "coded_turns": coded_turns,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def judge_path(path: Path, judge_model: str = JUDGE_MODEL, dry_run: bool = False) -> list[dict] | None:
    transcript = _load_transcript(path)
    turns = transcript["turns"]

    if transcript.get("outcome") == "parse_error":
        # The final turn is an {role, error, raw} record with no utterance/action —
        # negotiation.py's retries were exhausted, so this isn't a real completed
        # negotiation and there's nothing meaningful to FTA-code.
        print(f"Skipping {path.name}: outcome is parse_error, not a completed negotiation")
        return None

    print(f"Transcript: {path.name} | turns: {len(turns)} | judge model: {judge_model}")
    if dry_run:
        print("[dry run] No API calls made.")
        return None

    coded_turns = code_transcript(turns, judge_model=judge_model)
    output_path = _write_judged_transcript(path, transcript, coded_turns, judge_model)
    print(f"Saved judged transcript -> {output_path}")
    print(json.dumps(coded_turns, ensure_ascii=False, indent=2))
    return coded_turns


def _iter_transcript_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate for candidate in path.glob("*.json")
        if not candidate.name.endswith("_judged.json")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=Path(__file__).parent.parent / "results" / "transcripts",
        type=Path,
        help="Transcript JSON file or directory of transcript JSON files",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only judge the first N transcripts in a directory")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be judged, no API calls")
    parser.add_argument("--model", default=JUDGE_MODEL, help="Judge model slug to use")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of transcripts to judge concurrently (default {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    transcript_paths = _iter_transcript_paths(args.path)
    if args.limit is not None:
        transcript_paths = transcript_paths[:args.limit]

    if not transcript_paths:
        print(f"No transcript JSON files found at {args.path}")
        return

    if args.dry_run:
        print(f"Found {len(transcript_paths)} transcript(s) to judge")
        for transcript_path in transcript_paths:
            judge_path(transcript_path, judge_model=args.model, dry_run=True)
        return

    # resume-safe: skip transcripts that already have a judged output file
    pending = [p for p in transcript_paths if not _default_output_path(p).exists()]
    already_done = len(transcript_paths) - len(pending)
    print(f"Found {len(transcript_paths)} transcript(s) | already judged: {already_done} | "
          f"pending: {len(pending)} | concurrency: {args.workers} workers")

    failures: list[tuple[str, str]] = []
    with tqdm(total=len(transcript_paths), initial=already_done) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(judge_path, path, args.model, False): path
                for path in pending
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append((path.name, str(exc)))
                    print(f"FAILED {path.name}: {exc}")
                pbar.update(1)

    print(f"\nDone. {len(pending) - len(failures)}/{len(pending)} new transcripts judged.")
    if failures:
        print(f"{len(failures)} transcript(s) failed (re-run to retry, resume-safe):")
        for name, err in failures:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
