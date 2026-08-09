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

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Which provider each model name routes to
GEMINI_MODELS = {"gemini-3.6-flash", "gemini-3.6-flash-lite"}
OPENROUTER_MODELS = {"qwen/qwen3-coder:free", "meta-llama/llama-3.3-70b-instruct:free"}


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
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    gm = genai.GenerativeModel(model, system_instruction=system_prompt)
    chat_history = [
        {"role": "model" if t["role"] == self_role else "user", "parts": [t["content"]]}
        for t in history
    ]
    chat = gm.start_chat(history=chat_history)
    response = chat.send_message(
        "Continue the negotiation with your next turn.",
        generation_config={"temperature": temperature},
    )
    return response.text


def _call_openrouter(model: str, system_prompt: str, history: list[dict], self_role: str, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    messages = _history_to_messages(system_prompt, history, self_role)
    completion = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
    )
    return completion.choices[0].message.content


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
