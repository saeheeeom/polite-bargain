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
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"

TRANSLATE_PROMPT = """Translate the following product title and description into natural,
casual Korean, as they would appear in a Korean secondhand marketplace listing (like Danggeun
Market). Respond with ONLY a JSON object: {{"title_ko": "...", "description_ko": "..."}}

Title: {title}
Description: {description}
"""


# Free-tier cap observed for gemini-3.6-flash elsewhere in this project (see agents.py) —
# this script calls the API directly rather than through call_agent(), so it needs its
# own throttle to avoid tripping the same per-minute quota.
GEMINI_REQUESTS_PER_MINUTE_LIMIT = 5
GEMINI_MIN_SECONDS_BETWEEN_CALLS = 60 / GEMINI_REQUESTS_PER_MINUTE_LIMIT

_last_call_time = 0.0


def _throttle() -> None:
    import time

    global _last_call_time
    wait = _last_call_time + GEMINI_MIN_SECONDS_BETWEEN_CALLS - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.monotonic()


def translate_scenario(title: str, description: str, max_retries: int = 4) -> dict:
    import re
    import time
    from google import genai
    from google.genai import errors

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for attempt in range(max_retries):
        _throttle()
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
        except errors.ClientError as e:
            if getattr(e, "code", None) != 429 and "RESOURCE_EXHAUSTED" not in str(e):
                raise
            wait = GEMINI_MIN_SECONDS_BETWEEN_CALLS * (attempt + 1)
            print(f"  Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed to translate {title!r} after {max_retries} retries")


def _save(scenarios: list[dict]) -> None:
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        scenarios = json.load(f)

    for s in scenarios:
        if "title_ko" in s:
            continue  # already translated, resume-safe
        translated = translate_scenario(s["title"], s["description"])
        s["title_ko"] = translated["title_ko"]
        s["description_ko"] = translated["description_ko"]
        print(f"Translated: {s['title']} -> {s['title_ko']}")
        _save(scenarios)  # save after EVERY translation, not just at the end

    print(f"\nDone. Now do the spot-check pass: read through data/scenarios.json "
          f"and fix any title_ko/description_ko that reads unnaturally, "
          f"especially checking a couple per category as planned in the research statement.")


if __name__ == "__main__":
    main()
