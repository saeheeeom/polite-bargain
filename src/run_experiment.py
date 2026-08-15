"""
Main entry point — runs all scenarios x conditions x repetitions.

Usage:
    python -m src.run_experiment --dry-run   # estimate call count and time, no API calls
    python -m src.run_experiment              # actually run it
    python -m src.run_experiment --limit 2     # only run 2 scenarios (for smoke testing)
    python -m src.run_experiment --provider gemini  # switch back to Gemini for negotiations
    python -m src.run_experiment --workers 12  # run 12 negotiations concurrently (default 8)

Negotiations run concurrently via a thread pool (safe: each negotiation is
independent and writes its own uniquely-named transcript file, and the LLM
API calls are I/O-bound). Still resume-safe — already-completed
scenario/condition/rep files are skipped before submitting work.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from src.conditions import N_REPETITIONS, all_conditions
from src.negotiation import run_negotiation

DATA_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "transcripts"

AVG_TURNS_ESTIMATE = 9  # from CraigslistBargain's reported average dialogue length
DEFAULT_WORKERS = 8
DEFAULT_NEGOTIATION_MODELS = {
    "gemini": ("gemini-3.6-flash", "gemini-3.6-flash"),
    "openrouter": (
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-26b-a4b-it:free",
    ),
}


def load_scenarios(limit: int | None = None) -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        scenarios = json.load(f)
    return scenarios[:limit] if limit else scenarios


def estimate_calls(n_scenarios: int, n_conditions: int, n_repetitions: int) -> dict:
    n_negotiations = n_scenarios * n_conditions * n_repetitions
    negotiation_calls = n_negotiations * AVG_TURNS_ESTIMATE
    judge_calls = n_negotiations  # one judge call per finished transcript; this uses a separate non-Gemini model
    total = negotiation_calls + judge_calls
    return {
        "n_negotiations": n_negotiations,
        "negotiation_calls": negotiation_calls,
        "judge_calls": judge_calls,
        "total_calls": total,
    }


def resolve_negotiation_models(provider: str) -> tuple[str, str]:
    try:
        return DEFAULT_NEGOTIATION_MODELS[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {provider}") from exc


def _run_one(
    scenario_id: str, scenario: dict, condition, rep: int, out_path: Path,
    buyer_model: str, seller_model: str,
) -> None:
    result = run_negotiation(
        scenario=scenario, condition=condition,
        scenario_id=scenario_id, repetition=rep,
        buyer_model=buyer_model,
        seller_model=seller_model,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost, don't call any APIs")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N scenarios (for smoke testing)")
    parser.add_argument(
        "--provider",
        choices=sorted(DEFAULT_NEGOTIATION_MODELS),
        default="openrouter",
        help="Which provider to use for buyer/seller negotiation calls",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of negotiations to run concurrently (default {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(limit=args.limit)
    conditions = all_conditions()
    buyer_model, seller_model = resolve_negotiation_models(args.provider)

    est = estimate_calls(len(scenarios), len(conditions), N_REPETITIONS)
    print(f"Scenarios: {len(scenarios)} | Conditions: {len(conditions)} | Repetitions: {N_REPETITIONS}")
    print(f"Negotiation provider: {args.provider} ({buyer_model}) | Concurrency: {args.workers} workers")
    print(f"Estimated negotiations: {est['n_negotiations']}")
    print(f"Estimated total API calls: {est['total_calls']} "
          f"(~{est['negotiation_calls']} negotiation + {est['judge_calls']} judge)")

    if args.dry_run:
        print("\n[dry run] No API calls made.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for scenario_idx, scenario in enumerate(scenarios):
        scenario_id = f"scenario-{scenario_idx:02d}"
        for condition in conditions:
            for rep in range(N_REPETITIONS):
                out_path = RESULTS_DIR / f"{scenario_id}_{condition.id}_rep{rep}.json"
                if out_path.exists():
                    continue  # resume-safe: skip already-completed runs
                tasks.append((scenario_id, scenario, condition, rep, out_path))

    total = len(scenarios) * len(conditions) * N_REPETITIONS
    already_done = total - len(tasks)
    failures = []

    with tqdm(total=total, initial=already_done) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_one, scenario_id, scenario, condition, rep, out_path,
                    buyer_model, seller_model,
                ): (scenario_id, condition.id, rep)
                for scenario_id, scenario, condition, rep, out_path in tasks
            }
            for future in as_completed(futures):
                scenario_id, condition_id, rep = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append((scenario_id, condition_id, rep, str(exc)))
                    print(f"FAILED {scenario_id} | {condition_id} | rep {rep}: {exc}")
                pbar.update(1)

    print(f"\nDone. {len(tasks) - len(failures)}/{len(tasks)} new negotiations completed. "
          f"Transcripts saved to {RESULTS_DIR}")
    if failures:
        print(f"{len(failures)} negotiation(s) failed and were NOT saved (re-run to retry, resume-safe):")
        for scenario_id, condition_id, rep, err in failures:
            print(f"  {scenario_id} | {condition_id} | rep {rep}: {err}")


if __name__ == "__main__":
    main()
