"""
LLM API wrappers for the negotiating agents and the judge.

Both providers are called through a single `call_agent()` function so the rest
of the codebase doesn't care which provider is behind a given model name.

NOT YET TESTED against live APIs in this sandbox (no network access to the
provider endpoints here) — test with a single call before running the full
experiment. A minimal smoke test is at the bottom of this file.
"""
import json
import os
import re
import time
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from openai import RateLimitError


GEMINI_REQUESTS_PER_MINUTE_LIMIT = 5
GEMINI_MIN_SECONDS_BETWEEN_CALLS = 60 / GEMINI_REQUESTS_PER_MINUTE_LIMIT

_gemini_call_lock = Lock()
_gemini_next_allowed_time = 0.0


def _load_api_keys() -> None:
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / ".env",                 # standard project-level .env file if present
        project_root / ".env" / "config.yml",  # current project config file in this repo
        project_root / ".env" / "config.yaml",
        project_root / "config.yml",
    ]
    for path in candidates:
        if path.exists():
            if load_dotenv(dotenv_path=path, override=False):
                break


_load_api_keys()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Which provider each model name routes to
GEMINI_MODELS = {"gemini-3.6-flash", "gemini-3.6-flash-lite"}
OPENROUTER_MODELS = {
    "qwen/qwen3-coder:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
}


def call_agent(model: str, system_prompt: str, history: list[dict], self_role: str, temperature: float = 0.2) -> str:
    """
    history: list of {"role": "buyer"|"seller", "content": str} in chronological order.
    self_role: "buyer" or "seller" — whichever role THIS call is generating a turn for.
    Returns the raw text response (caller parses it with parse_agent_response).
    """
    if model in GEMINI_MODELS:
        return _call_gemini(model, system_prompt, history, self_role, temperature)
    elif model in OPENROUTER_MODELS:
        return _call_openrouter(model, system_prompt, history, self_role, temperature)
    else:
        raise ValueError(f"Unknown model: {model}. Add it to GEMINI_MODELS or OPENROUTER_MODELS.")


def _history_to_messages(system_prompt: str, history: list[dict], self_role: str) -> list[dict]:
    """Converts our internal history format to OpenAI-style chat messages,
    from the perspective of `self_role` (so the other party's turns become "user"
    and this agent's own past turns become "assistant")."""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        role = "assistant" if turn["role"] == self_role else "user"
        messages.append({"role": role, "content": turn["content"]})
    return messages


def _call_gemini(model: str, system_prompt: str, history: list[dict], self_role: str, temperature: float) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [
        types.Content(
            role=("model" if t["role"] == self_role else "user"),
            parts=[types.Part(text=t["content"])],
        )
        for t in history
    ]
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text="Continue the negotiation with your next turn.")],
    ))

    def _wait_for_gemini_slot() -> None:
        global _gemini_next_allowed_time
        with _gemini_call_lock:
            now = time.monotonic()
            sleep_for = _gemini_next_allowed_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            _gemini_next_allowed_time = time.monotonic() + GEMINI_MIN_SECONDS_BETWEEN_CALLS

    def _is_rate_limit_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        return status_code == 429 or "RESOURCE_EXHAUSTED" in str(exc)

    last_error: Exception | None = None
    for attempt in range(6):
        _wait_for_gemini_slot()
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                ),
            )
            return response.text
        except Exception as exc:  # google.genai raises provider-specific errors here
            last_error = exc
            if not _is_rate_limit_error(exc) or attempt == 5:
                raise
            retry_delay = min(30.0, GEMINI_MIN_SECONDS_BETWEEN_CALLS * (attempt + 1))
            time.sleep(retry_delay)

    raise last_error if last_error is not None else RuntimeError("Gemini request failed without a response")


def _call_openrouter(model: str, system_prompt: str, history: list[dict], self_role: str, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    messages = _history_to_messages(system_prompt, history, self_role)

    def _retry_delay_from_error(exc: Exception, fallback_seconds: float) -> float:
        retry_after = re.search(r"retry_after_seconds(?:_raw)?['\"]?:\s*(\d+)", str(exc))
        if retry_after:
            return float(retry_after.group(1))
        return fallback_seconds

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            completion = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
            )
            return completion.choices[0].message.content
        except RateLimitError as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(_retry_delay_from_error(exc, fallback_seconds=5.0 * (attempt + 1)))

    raise last_error if last_error is not None else RuntimeError("OpenRouter request failed without a response")


AGENT_RESPONSE_INSTRUCTIONS = """
Respond with ONLY a JSON object in this exact shape, nothing else:
{
  "utterance": "<what you say out loud>",
  "action": "<one of: offer, accept, reject, counter-offer>",
  "price": <number, or null if the action has no associated price>
}
"""


def parse_agent_response(raw: str) -> dict:
    """Extracts the JSON object from a raw model response, tolerating markdown
    code fences or stray text around it."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw!r}")
    obj = json.loads(match.group(0))
    for key in ("utterance", "action", "price"):
        if key not in obj:
            raise ValueError(f"Missing key '{key}' in parsed response: {obj}")
    return obj


if __name__ == "__main__":
    # Minimal smoke test — run this first, locally, before anything else.
    # Requires GEMINI_API_KEY to be set in .env.
    test_system_prompt = (
        "You are a buyer in a price negotiation for a used bicycle listed at $100. "
        "Your target price is $70." + AGENT_RESPONSE_INSTRUCTIONS
    )
    raw = call_agent("gemini-3.6-flash", test_system_prompt, history=[], self_role="buyer")
    print("Raw response:", raw)
    parsed = parse_agent_response(raw)
    print("Parsed:", parsed)
