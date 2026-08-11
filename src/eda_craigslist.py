"""EDA helper for the CraigslistBargain scenarios + transcripts.

Usage:
    python -m src.eda_craigslist [--scenarios data/scenarios.json] [--transcripts results/transcripts]

The script will:
- try to load dataset metadata from Hugging Face (`stanfordnlp/craigslist_bargains`) if `datasets` is available
- load local `data/scenarios.json` and print summary statistics
- scan `results/transcripts` for negotiation results (if any) and compute outcome/turn stats
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import statistics
from typing import Any


def try_load_hf(name: str) -> Any | None:
    try:
        from datasets import load_dataset

        print(f"Attempting to load HuggingFace dataset: {name}...")
        ds = load_dataset(name)
        return ds
    except Exception as e:
        print(f"Could not load HF dataset ({name}): {e}")
        return None


def load_local_scenarios(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_scenarios(scenarios: list[dict]) -> dict:
    n = len(scenarios)
    cats = [s.get("category", "<none>") for s in scenarios]
    cat_counts = Counter(cats)

    def num(field: str):
        vals = [s[field] for s in scenarios if s.get(field) is not None]
        if not vals:
            return {}
        return {
            "count": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    listing_stats = num("listing_price")
    buyer_stats = num("buyer_target")
    seller_stats = num("seller_target")

    # price gap: listing - buyer_target
    gaps = [s["listing_price"] - s["buyer_target"] for s in scenarios if s.get("listing_price") is not None and s.get("buyer_target") is not None]
    gap_stats = {}
    if gaps:
        gap_stats = {
            "count": len(gaps),
            "mean": statistics.mean(gaps),
            "median": statistics.median(gaps),
            "min": min(gaps),
            "max": max(gaps),
        }

    return {
        "n_scenarios": n,
        "n_categories": len(cat_counts),
        "category_counts": cat_counts.most_common(),
        "listing_price": listing_stats,
        "buyer_target": buyer_stats,
        "seller_target": seller_stats,
        "listing_minus_buyer_gap": gap_stats,
    }


def load_transcripts(dirpath: Path) -> list[dict]:
    if not dirpath.exists():
        return []
    files = [p for p in dirpath.iterdir() if p.suffix == ".json"]
    results = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                results.append(json.load(fh))
        except Exception:
            continue
    return results


def summarize_transcripts(transcripts: list[dict]) -> dict:
    if not transcripts:
        return {"n_transcripts": 0}

    outcomes = Counter(t.get("outcome", "unknown") for t in transcripts)
    n_turns = [t.get("n_turns", 0) for t in transcripts]
    avg_turns = statistics.mean(n_turns) if n_turns else 0

    by_outcome = {}
    for o in outcomes:
        subset = [t for t in transcripts if t.get("outcome") == o]
        turns = [s.get("n_turns", 0) for s in subset]
        by_outcome[o] = {
            "count": len(subset),
            "avg_turns": statistics.mean(turns) if turns else 0,
        }

    example_dialog = None
    # pick first transcript with turns to show as example
    for t in transcripts:
        if t.get("turns"):
            example_dialog = t
            break

    return {
        "n_transcripts": len(transcripts),
        "outcomes": outcomes.most_common(),
        "avg_turns_all": avg_turns,
        "by_outcome": by_outcome,
        "example_dialog": example_dialog,
    }


def print_example_dialog(example: dict) -> None:
    if not example:
        print("No example dialog available.")
        return
    print("\nExample dialogue (first available transcript):")
    print(f"scenario_id: {example.get('scenario_id')} | condition: {example.get('condition_id')} | repetition: {example.get('repetition')}")
    for idx, turn in enumerate(example.get("turns", []), 1):
        role = turn.get("role")
        utt = turn.get("utterance")
        action = turn.get("action")
        price = turn.get("price")
        line = f"{idx:02d}. {role}: {utt}"
        extras = []
        if action:
            extras.append(f"action={action}")
        if price is not None:
            extras.append(f"price={price}")
        if extras:
            line += "  (" + ", ".join(extras) + ")"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="data/scenarios.json")
    parser.add_argument("--transcripts", default="results/transcripts")
    parser.add_argument("--hf-name", default="stanfordnlp/craigslist_bargains")
    args = parser.parse_args()

    hf = try_load_hf(args.hf_name)
    if hf is not None:
        print("HuggingFace dataset loaded — configs/splits:" , getattr(hf, "keys", lambda: None)())
    else:
        print("HuggingFace dataset not available locally.")

    scen_path = Path(args.scenarios)
    if not scen_path.exists():
        print(f"Local scenarios file not found: {scen_path}")
        return

    scenarios = load_local_scenarios(scen_path)
    summary = summarize_scenarios(scenarios)
    print("\n=== Scenarios summary ===")
    print(f"Total scenarios: {summary['n_scenarios']}")
    print(f"Distinct categories: {summary['n_categories']}")
    print("Top categories:")
    for cat, cnt in summary["category_counts"][:10]:
        print(f"  - {cat}: {cnt}")
    print("Listing price stats:", summary["listing_price"])
    print("Buyer target stats:", summary["buyer_target"])
    print("Seller target stats:", summary["seller_target"])
    print("Listing - buyer gap stats:", summary["listing_minus_buyer_gap"])

    # transcripts
    tdir = Path(args.transcripts)
    transcripts = load_transcripts(tdir)
    t_summary = summarize_transcripts(transcripts)
    print("\n=== Transcripts summary ===")
    if t_summary.get("n_transcripts", 0) == 0:
        print("No transcripts found in", tdir)
    else:
        print(f"Total transcripts: {t_summary['n_transcripts']}")
        print("Outcomes:")
        for o, c in t_summary["outcomes"]:
            print(f"  - {o}: {c}")
        print("Average turns (all):", t_summary["avg_turns_all"])
        print("By outcome:")
        for o, info in t_summary["by_outcome"].items():
            print(f"  - {o}: count={info['count']} avg_turns={info['avg_turns']}")
        print_example_dialog(t_summary.get("example_dialog"))


if __name__ == "__main__":
    main()
