"""One negotiation against an OpenAI-compatible server (vLLM), asynchronously.

The turn loop mirrors src/negotiation.py, and the prompts, message format and
response parser are imported from the paper code rather than copied, so the
open-model runs use exactly the paper's inputs. What's new is what gets
recorded: a deterministic seed per request, token usage, latency and retries.
"""
from __future__ import annotations

import asyncio
import hashlib
import time

import openai

from harness.config import RunConfig
from src.agents import _history_to_messages, parse_agent_response
from src.conditions import Condition
from src.negotiation import _last_offered_price, build_system_prompt

MAX_TRANSPORT_RETRIES = 5


def request_seed(base_seed: int, transcript_id: str, turn: int, attempt: int) -> int:
    """Seed for one generation call, fixed by where the call sits in the design.

    Seeding per request (not per process) means a rerun gives each turn the same
    seed no matter how requests were interleaved across concurrent negotiations.
    """
    payload = f"{base_seed}|{transcript_id}|{turn}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


async def _generate(client: openai.AsyncOpenAI, config: RunConfig, messages: list[dict], seed: int) -> tuple[str, dict]:
    """One completion, retrying only transport failures (server busy or restarting)."""
    for attempt in range(MAX_TRANSPORT_RETRIES + 1):
        started = time.perf_counter()
        try:
            completion = await client.chat.completions.create(
                model=config.model.hf_id,
                messages=messages,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_tokens,
                seed=seed,
                extra_body={"chat_template_kwargs": config.model.chat_template_kwargs},
            )
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
            if attempt == MAX_TRANSPORT_RETRIES:
                raise
            await asyncio.sleep(2**attempt)
            continue
        usage = completion.usage
        return completion.choices[0].message.content or "", {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "latency_s": round(time.perf_counter() - started, 4),
            "finish_reason": completion.choices[0].finish_reason,
            "transport_retries": attempt,
        }
    raise AssertionError("unreachable")


async def run_negotiation(
    client: openai.AsyncOpenAI,
    config: RunConfig,
    scenario: dict,
    scenario_id: str,
    condition: Condition,
    repetition: int,
) -> dict:
    transcript_id = f"{scenario_id}_{condition.id}_rep{repetition}"
    prompts = {role: build_system_prompt(role, scenario, condition) for role in ("buyer", "seller")}
    turns: list[dict] = []
    calls: list[dict] = []
    history: list[dict] = []
    outcome, final_price = "max_turns_reached", None
    role = "seller"  # seller opens, as in the paper

    for turn_num in range(config.design.max_turns):
        messages = _history_to_messages(prompts[role], history, self_role=role)
        parsed, error, raw = None, None, ""
        for attempt in range(config.design.max_parse_retries):
            seed = request_seed(config.generation.base_seed, transcript_id, turn_num, attempt)
            raw, call = await _generate(client, config, messages, seed)
            calls.append({"turn": turn_num, "attempt": attempt, "seed": seed, **call})
            try:
                parsed = parse_agent_response(raw)
                break
            except ValueError as exc:  # json.JSONDecodeError is a ValueError
                error = exc

        if parsed is None:
            outcome = "parse_error"
            turns.append({"role": role, "error": str(error), "raw": raw})
            break

        turns.append(
            {"role": role, "utterance": parsed["utterance"], "action": parsed["action"], "price": parsed["price"]}
        )
        history.append({"role": role, "content": parsed["utterance"]})

        if parsed["action"] == "accept":
            outcome, final_price = "agreed", _last_offered_price(turns)
            break
        if parsed["action"] == "reject" and turn_num > 0:
            outcome = "abandoned"
            break
        role = "buyer" if role == "seller" else "seller"

    # Same top-level schema as the paper's transcripts, so harness.analysis reads both.
    return {
        "scenario_id": scenario_id,
        "condition_id": condition.id,
        "repetition": repetition,
        "turns": turns,
        "outcome": outcome,
        "final_price": final_price,
        "n_turns": len(turns) if outcome != "parse_error" else 0,
        "calls": calls,
    }
