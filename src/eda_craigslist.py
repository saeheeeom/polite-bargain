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


def summarize_hf_split(split_data: Any) -> dict:
    rows = list(split_data)
    if not rows:
        return {"rows": 0}

    categories: list[str] = []
    prices: list[float] = []
    title_lengths: list[float] = []
    description_lengths: list[float] = []
    turn_counts: list[float] = []
    utterance_lengths: list[float] = []
    intent_counts: Counter[str] = Counter()
    accept_count = 0

    for row in rows:
        items = row.get("items", {}) or {}
        item_categories = items.get("Category") or []
        if isinstance(item_categories, list):
            categories.extend([c for c in item_categories if c])
        else:
            categories.extend([str(item_categories)])

        item_prices = items.get("Price") or []
        if isinstance(item_prices, list):
            prices.extend([float(p) for p in item_prices if p is not None])

        titles = items.get("Title") or []
        if isinstance(titles, list):
            title_lengths.extend([len(str(t)) for t in titles if t is not None])

        descriptions = items.get("Description") or []
        if isinstance(descriptions, list):
            description_lengths.extend([len(str(d)) for d in descriptions if d is not None])

        agent_turns = row.get("agent_turn") or []
        if isinstance(agent_turns, list):
            turn_counts.append(len(agent_turns))

        utterances = row.get("utterance") or []
        if isinstance(utterances, list):
            utterance_lengths.extend([len(str(u)) for u in utterances if u is not None])

        dialogue_acts = row.get("dialogue_acts") or {}
        intents = dialogue_acts.get("intent") or []
        if isinstance(intents, list):
            for intent in intents:
                if intent:
                    intent_counts[str(intent)] += 1
            if "accept" in intents:
                accept_count += 1

    def summarize_numbers(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "min": min(values),
            "max": max(values),
        }

    return {
        "rows": len(rows),
        "categories": Counter(categories).most_common(10),
        "price_stats": summarize_numbers(prices),
        "title_length_stats": summarize_numbers(title_lengths),
        "description_length_stats": summarize_numbers(description_lengths),
        "turn_count_stats": summarize_numbers(turn_counts),
        "utterance_length_stats": summarize_numbers(utterance_lengths),
        "intent_counts": intent_counts.most_common(10),
        "accept_rate": round(accept_count / len(rows), 4) if rows else 0.0,
    }


def print_hf_dataset_summary(ds: Any) -> None:
    if ds is None:
        return

    print("Stanford dataset summary (Hugging Face):")
    if isinstance(ds, dict):
        for split_name, split_data in ds.items():
            try:
                summary = summarize_hf_split(split_data)
            except Exception as exc:
                print(f"  - {split_name}: could not summarize ({exc})")
                continue

            print(f"  - {split_name}: {summary['rows']} rows")
            print(f"    categories: {summary['categories']}")
            print(f"    price stats: {summary['price_stats']}")
            print(f"    title length stats: {summary['title_length_stats']}")
            print(f"    description length stats: {summary['description_length_stats']}")
            print(f"    turn count stats: {summary['turn_count_stats']}")
            print(f"    utterance length stats: {summary['utterance_length_stats']}")
            print(f"    top dialogue intents: {summary['intent_counts']}")
            print(f"    accept-rate proxy: {summary['accept_rate']}")

            try:
                example = list(split_data)[0]
                print("    example row:")
                print(f"      - items: {example.get('items', {}).get('Title', [])}")
                print(f"      - utterances: {example.get('utterance', [])}")
                print(f"      - intents: {example.get('dialogue_acts', {}).get('intent', [])}")
            except Exception:
                pass
    else:
        print(f"  loaded object type: {type(ds).__name__}")


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


def print_example_dialog(example: dict[str, Any] | None) -> None:
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
        print("HuggingFace dataset loaded — configs/splits:", list(getattr(hf, "keys", lambda: [])()))
        print_hf_dataset_summary(hf)
        print("Note: this dataset is bargaining-style dialogue data, but it is not the same as the negotiation transcript files your experiment will generate.")
    else:
        print("HuggingFace dataset not available locally.")

    scen_path = Path(args.scenarios)
    if not scen_path.exists():
        print(f"Local scenarios file not found: {scen_path}")
        return

    scenarios = load_local_scenarios(scen_path)
    summary = summarize_scenarios(scenarios)
    print("\n=== Local project scenario summary (your 12-item sample) ===")
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
