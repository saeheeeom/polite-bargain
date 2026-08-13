"""
Translates each sampled scenario's title/description into Korean and adds
`title_ko` / `description_ko` fields to data/scenarios.json in place.

This is a GAP that was missing from the original skeleton — negotiation.py's
build_system_prompt() was silently using the English title/description even
in the "ko" language condition, which would have quietly broken the whole
point of the language manipulation. Run this BEFORE run_experiment.py.

This uses the Gemini API itself for translation (no separate translation API
needed), then you do the lightweight spot-check pass described in the
research statement (read through a handful per category, fix anything that
reads unnaturally).
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
for candidate in (
    project_root / ".env",
    project_root / ".env" / "config.yml",
    project_root / ".env" / "config.yaml",
    project_root / "config.yml",
):
    if candidate.exists() and load_dotenv(dotenv_path=candidate, override=False):
        break

DATA_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"

TRANSLATE_PROMPT = """Translate the following product title and description into natural,
casual Korean, as they would appear in a Korean secondhand marketplace listing (like Danggeun
Market). Respond with ONLY a JSON object: {{"title_ko": "...", "description_ko": "..."}}

Title: {title}
Description: {description}
"""


def translate_scenario(title: str, description: str, max_retries: int = 4) -> dict:
    import re
    import time
    from google import genai
    from google.genai import errors

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=TRANSLATE_PROMPT.format(title=title, description=description),
            )
            match = re.search(r"\{.*\}", response.text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in translation response: {response.text!r}")
            return json.loads(match.group(0))
        except errors.ServerError as e:
            # 503 = transient overload on Google's end, not a bug in this code — retry with backoff
            wait = 2 ** attempt  # 1, 2, 4, 8 seconds
            print(f"  Server error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed to translate {title!r} after {max_retries} retries")


def _save(scenarios: list[dict]) -> None:
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)


def _load_scenarios() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def translate_pending_scenarios(
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    scenarios = _load_scenarios()
    pending_indices = [i for i, s in enumerate(scenarios) if force or "title_ko" not in s]
    selected_indices = pending_indices[:limit] if limit is not None else pending_indices

    print(
        f"Scenarios total: {len(scenarios)} | already translated: {len(scenarios) - len(pending_indices)} | "
        f"pending: {len(pending_indices)} | selected: {len(selected_indices)}"
    )

    if dry_run:
        print("[dry run] No API calls made.")
        return 0

    translated_count = 0
    for index in selected_indices:
        scenario = scenarios[index]
        translated = translate_scenario(scenario["title"], scenario["description"])
        scenario["title_ko"] = translated["title_ko"]
        scenario["description_ko"] = translated["description_ko"]
        print(f"Translated: {scenario['title']} -> {scenario['title_ko']}")
        _save(scenarios)  # save after EVERY translation, not just at the end
        translated_count += 1

    return translated_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only translate the first N untranslated scenarios")
    parser.add_argument("--dry-run", action="store_true", help="Report how many scenarios would be translated, no API calls")
    parser.add_argument("--force", action="store_true", help="Translate the selected scenarios even if title_ko already exists")
    args = parser.parse_args()

    translated_count = translate_pending_scenarios(limit=args.limit, dry_run=args.dry_run, force=args.force)

    if args.dry_run:
        return

    print(f"\nDone. Translated {translated_count} scenario(s). Now do the spot-check pass: read through data/scenarios.json "
          f"and fix any title_ko/description_ko that reads unnaturally, "
          f"especially checking a couple per category as planned in the research statement.")


if __name__ == "__main__":
    main()
