"""
Runs one full negotiation between a buyer agent and a seller agent, given a
scenario and a Condition (language, urgency, persona).
"""
from dataclasses import dataclass, field

from src.agents import AGENT_RESPONSE_INSTRUCTIONS, call_agent, parse_agent_response
from src.conditions import Condition
from src.translations import get_persona_text, get_urgency_text

MAX_TURNS = 10

BASE_PROMPT = {
    "en": {
        "buyer": "You are negotiating to buy: {title} — {description}. Listing price: ${price}. Your target price: ${target}.",
        "seller": "You are negotiating to sell: {title} — {description}. Listing price: ${price}. Your target price: ${target}.",
    },
    "ko": {
        "buyer": "당신은 다음 물건을 구매하기 위해 협상 중입니다: {title} — {description}. 정가: ${price}. 목표 가격: ${target}.",  # TODO: verify Korean
        "seller": "당신은 다음 물건을 판매하기 위해 협상 중입니다: {title} — {description}. 정가: ${price}. 목표 가격: ${target}.",  # TODO: verify Korean
    },
}


@dataclass
class NegotiationResult:
    scenario_id: str
    condition_id: str
    repetition: int
    turns: list[dict] = field(default_factory=list)  # each: {role, utterance, action, price}
    outcome: str = "abandoned"  # "agreed" | "abandoned" | "max_turns_reached"
    final_price: float | None = None
    n_turns: int = 0


def build_system_prompt(role: str, scenario: dict, condition: Condition) -> str:
    lang = condition.language
    target = scenario["buyer_target"] if role == "buyer" else scenario["seller_target"]

    if lang == "ko":
        if "title_ko" not in scenario:
            raise ValueError(
                f"Scenario {scenario.get('title')!r} has no Korean translation yet — "
                f"run `python -m src.translate_scenarios` before running Korean-condition negotiations."
            )
        title = scenario["title_ko"]
        description = scenario["description_ko"]
    else:
        title = scenario["title"]
        description = scenario["description"]

    prompt = BASE_PROMPT[lang][role].format(
        title=title, description=description,
        price=scenario["listing_price"], target=target,
    )
    if role == "seller":
        urgency_text = get_urgency_text(condition.urgency, lang)
        if urgency_text:
            prompt += " " + urgency_text
    if role == "buyer":
        persona_text = get_persona_text(condition.persona, lang)
        if persona_text:
            prompt += " " + persona_text
    prompt += "\n\n" + AGENT_RESPONSE_INSTRUCTIONS
    return prompt


def run_negotiation(
    scenario: dict,
    condition: Condition,
    scenario_id: str,
    repetition: int,
    buyer_model: str = "gemini-3.6-flash",
    seller_model: str = "gemini-3.6-flash",
) -> NegotiationResult:
    buyer_prompt = build_system_prompt("buyer", scenario, condition)
    seller_prompt = build_system_prompt("seller", scenario, condition)

    result = NegotiationResult(scenario_id=scenario_id, condition_id=condition.id, repetition=repetition)
    history: list[dict] = []

    # Seller usually opens in CraigslistBargain-style negotiations; adjust if you want buyer to open.
    current_role, current_model, current_prompt = "seller", seller_model, seller_prompt

    for turn_num in range(MAX_TURNS):
        raw = call_agent(current_model, current_prompt, history, self_role=current_role)
        try:
            parsed = parse_agent_response(raw)
        except ValueError as e:
            # Log and treat as an abandoned negotiation rather than crashing the whole run
            result.outcome = "parse_error"
            result.turns.append({"role": current_role, "error": str(e), "raw": raw})
            return result

        turn_record = {
            "role": current_role,
            "utterance": parsed["utterance"],
            "action": parsed["action"],
            "price": parsed["price"],
        }
        result.turns.append(turn_record)
        history.append({"role": current_role, "content": parsed["utterance"]})

        if parsed["action"] == "accept":
            result.outcome = "agreed"
            # final price is whatever was on the table — the price from the PRIOR turn
            result.final_price = _last_offered_price(result.turns)
            break
        if parsed["action"] == "reject" and turn_num > 0:
            # Treat an explicit reject with no counter as walking away.
            # (If your agents always counter instead of flatly rejecting, this branch
            # may rarely fire — that's fine, just don't rely on it as the only stop condition.)
            result.outcome = "abandoned"
            break

        # switch turns
        if current_role == "seller":
            current_role, current_model, current_prompt = "buyer", buyer_model, buyer_prompt
        else:
            current_role, current_model, current_prompt = "seller", seller_model, seller_prompt

    else:
        result.outcome = "max_turns_reached"

    result.n_turns = len(result.turns)
    return result


def _last_offered_price(turns: list[dict]) -> float | None:
    for t in reversed(turns):
        if t.get("price") is not None:
            return t["price"]
    return None
