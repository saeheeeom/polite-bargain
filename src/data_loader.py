"""
Loads stanfordnlp/craigslist_bargains from Hugging Face and samples scenarios
for the experiment.

NOTE: this needs live internet access to huggingface.co, which was not available
in the sandbox this skeleton was built in. Run this locally — it hasn't been
tested against the live dataset yet. The field names below are taken directly
from the dataset's README (confirmed via HF docs), so they should be correct,
but do a quick `print(dataset[0])` sanity check the first time you run this.

Known schema (from HF docs):
    agent_info: {Role: [...], Target: [...], Bottomline: [...]}
    items: {Category, Title, Description, Price}
    dialogue_acts: {intent: [...], price: [...]}
    utterance: [...]
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

from src.conditions import N_SCENARIOS

CATEGORIES = ["housing", "furniture", "car", "bike", "phone", "electronics"]
OUT_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"

SEED = 42  # fixed for reproducibility


def sample_scenarios(n_scenarios: int = N_SCENARIOS, seed: int = SEED) -> list[dict]:
    """
    Samples n_scenarios from CraigslistBargain, spread as evenly as possible
    across the 6 item categories.
    """
    random.seed(seed)
    ds = load_dataset("stanfordnlp/craigslist_bargains", split="train")

    # TODO once run locally: confirm `items.Category` is lowercase and matches
    # CATEGORIES exactly — adjust the filter below if not.
    by_category: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
    for row in ds:
        cat = row["items"]["Category"][0].lower()  # items fields are lists (buyer+seller share the item)
        if cat in by_category:
            by_category[cat].append(row)

    per_category = max(1, n_scenarios // len(CATEGORIES))
    sampled = []
    for cat, rows in by_category.items():
        if not rows:
            print(f"WARNING: no rows found for category '{cat}' — check category name spelling")
            continue
        k = min(per_category, len(rows))
        sampled.extend(random.sample(rows, k))

    # top up / trim to exactly n_scenarios if category counts didn't divide evenly
    random.shuffle(sampled)
    sampled = sampled[:n_scenarios]

    scenarios = []
    for row in sampled:
        scenarios.append({
            "category": row["items"]["Category"][0],
            "title": row["items"]["Title"][0],
            "description": row["items"]["Description"][0],
            "listing_price": row["items"]["Price"][0],
            "buyer_target": _get_target(row, "buyer"),
            "seller_target": _get_target(row, "seller"),
        })
    return scenarios


def _get_target(row: dict, role: str) -> float:
    roles = row["agent_info"]["Role"]
    targets = row["agent_info"]["Target"]
    idx = roles.index(role)
    return targets[idx]


def main():
    scenarios = sample_scenarios()
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)
    print(f"Sampled {len(scenarios)} scenarios -> {OUT_PATH}")
    by_cat = {}
    for s in scenarios:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
    print("Breakdown by category:", by_cat)


if __name__ == "__main__":
    main()
