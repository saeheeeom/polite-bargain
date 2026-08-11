"""
Main entry point — runs all scenarios x conditions x repetitions.

Usage:
    python -m src.run_experiment --dry-run   # estimate call count and time, no API calls
    python -m src.run_experiment              # actually run it
    python -m src.run_experiment --limit 2     # only run 2 scenarios (for smoke testing)
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from src.conditions import N_REPETITIONS, all_conditions
from src.negotiation import run_negotiation

DATA_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "transcripts"

AVG_TURNS_ESTIMATE = 9  # from CraigslistBargain's reported average dialogue length
GEMINI_REQUESTS_PER_MINUTE_LIMIT = 5  # observed live limit for gemini-3.6-flash on the free tier in this project


def load_scenarios(limit: int | None = None) -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        scenarios = json.load(f)
    return scenarios[:limit] if limit else scenarios


def estimate_calls(n_scenarios: int, n_conditions: int, n_repetitions: int) -> dict:
    n_negotiations = n_scenarios * n_conditions * n_repetitions
    negotiation_calls = n_negotiations * AVG_TURNS_ESTIMATE
    judge_calls = n_negotiations  # one judge call per finished transcript; this uses a separate non-Gemini model
    total = negotiation_calls + judge_calls
    # Gemini is rate-limited per minute here, so this is a minimum serialized wall-time estimate.
    est_minutes_gemini = negotiation_calls / GEMINI_REQUESTS_PER_MINUTE_LIMIT
    return {
        "n_negotiations": n_negotiations,
        "negotiation_calls": negotiation_calls,
        "judge_calls": judge_calls,
        "total_calls": total,
        "gemini_rpm_limit": GEMINI_REQUESTS_PER_MINUTE_LIMIT,
        "est_minutes_on_gemini_free_tier": round(est_minutes_gemini, 1),
        "est_hours_on_gemini_free_tier": round(est_minutes_gemini / 60, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost, don't call any APIs")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N scenarios (for smoke testing)")
    args = parser.parse_args()

    scenarios = load_scenarios(limit=args.limit)
    conditions = all_conditions()

    est = estimate_calls(len(scenarios), len(conditions), N_REPETITIONS)
    print(f"Scenarios: {len(scenarios)} | Conditions: {len(conditions)} | Repetitions: {N_REPETITIONS}")
    print(f"Estimated negotiations: {est['n_negotiations']}")
    print(f"Estimated total API calls: {est['total_calls']} "
          f"(~{est['negotiation_calls']} negotiation + {est['judge_calls']} judge)")
    print(f"Estimated Gemini free-tier cap used for planning: {est['gemini_rpm_limit']} requests/minute")
    print(f"Estimated minimum Gemini wall time: {est['est_minutes_on_gemini_free_tier']} minutes "
            f"(~{est['est_hours_on_gemini_free_tier']} hours) for negotiation calls only")

    if args.dry_run:
        print("\n[dry run] No API calls made.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(scenarios) * len(conditions) * N_REPETITIONS
    with tqdm(total=total) as pbar:
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_id = f"scenario-{scenario_idx:02d}"
            for condition in conditions:
                for rep in range(N_REPETITIONS):
                    out_path = RESULTS_DIR / f"{scenario_id}_{condition.id}_rep{rep}.json"
                    if out_path.exists():
                        pbar.update(1)
                        continue  # resume-safe: skip already-completed runs
                    result = run_negotiation(
                        scenario=scenario, condition=condition,
                        scenario_id=scenario_id, repetition=rep,
                    )
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
                    pbar.update(1)

    print(f"\nDone. Transcripts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
