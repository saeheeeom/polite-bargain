"""The async turn loop, against a scripted fake server instead of vLLM."""
import asyncio
import json
from types import SimpleNamespace

from harness.config import ModelConfig, RunConfig
from harness.negotiation import request_seed, run_negotiation
from src.conditions import Condition

SCENARIO = {
    "title": "Bike", "description": "Road bike", "listing_price": 200,
    "buyer_target": 100, "seller_target": 200,
}
CONFIG = RunConfig(name="test", model=ModelConfig(hf_id="fake/model", revision="abc"))


class FakeClient:
    """Returns scripted replies in order and records each request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def reply(action, price, utterance="..."):
    return json.dumps({"utterance": utterance, "action": action, "price": price})


def negotiate(replies, condition=Condition("en", "none", "neutral")):
    client = FakeClient(replies)
    result = asyncio.run(run_negotiation(client, CONFIG, SCENARIO, "scenario-00", condition, 0))
    return result, client


def test_agreement_takes_last_offered_price():
    result, _ = negotiate([reply("offer", 200), reply("counter-offer", 150), reply("accept", 150)])
    assert result["outcome"] == "agreed"
    assert result["final_price"] == 150
    assert result["n_turns"] == 3
    assert [turn["role"] for turn in result["turns"]] == ["seller", "buyer", "seller"]


def test_reject_after_first_turn_abandons():
    result, _ = negotiate([reply("offer", 200), reply("reject", None)])
    assert result["outcome"] == "abandoned"
    assert result["final_price"] is None


def test_malformed_reply_is_retried_with_a_new_seed():
    result, client = negotiate(["not json", reply("offer", 200), reply("accept", 200)])
    assert result["outcome"] == "agreed"
    assert [call["attempt"] for call in result["calls"]] == [0, 1, 0]
    assert client.requests[0]["seed"] != client.requests[1]["seed"]


def test_three_parse_failures_end_in_parse_error():
    result, _ = negotiate(["bad", "bad", "bad"])
    assert result["outcome"] == "parse_error"
    assert result["n_turns"] == 0
    assert "error" in result["turns"][-1]


def test_turn_cap():
    replies = [reply("offer", 200)] + [reply("counter-offer", 150)] * 15
    result, _ = negotiate(replies)
    assert result["outcome"] == "max_turns_reached"
    assert result["n_turns"] == 16


def test_messages_are_from_the_speakers_perspective():
    _, client = negotiate([reply("offer", 200, "Seller says hi"), reply("accept", 200)])
    buyer_messages = client.requests[1]["messages"]
    assert buyer_messages[0]["role"] == "system"
    assert buyer_messages[1] == {"role": "user", "content": "Seller says hi"}


def test_korean_condition_uses_korean_prompt():
    scenario = {**SCENARIO, "title_ko": "자전거", "description_ko": "로드 자전거"}
    client = FakeClient([reply("offer", 200), reply("accept", 200)])
    asyncio.run(run_negotiation(client, CONFIG, scenario, "scenario-00", Condition("ko", "none", "neutral"), 0))
    assert "자전거" in client.requests[0]["messages"][0]["content"]


def test_request_seed_is_stable_and_position_dependent():
    assert request_seed(42, "a", 0, 0) == request_seed(42, "a", 0, 0)
    assert len({request_seed(42, "a", t, k) for t in range(4) for k in range(3)}) == 12


def _config(message_format):
    from harness.config import ProtocolConfig
    return RunConfig(name="t", model=CONFIG.model, protocol=ProtocolConfig(message_format=message_format))


def _buyer_messages(message_format):
    client = FakeClient([reply("offer", 200, "Seller says hi"), reply("counter-offer", 150, "Buyer"), reply("accept", 150)])
    asyncio.run(run_negotiation(client, _config(message_format), SCENARIO, "scenario-00",
                                Condition("en", "none", "neutral"), 0))
    return client.requests


def test_continue_format_appends_turn_request():
    requests = _buyer_messages("continue")
    assert requests[0]["messages"][-1] == {"role": "user", "content": "Continue the negotiation with your next turn."}
    assert requests[1]["messages"][-1]["content"] == "Continue the negotiation with your next turn."


def test_continue_json_shows_own_turns_as_json_and_partner_as_text():
    seller_turn2 = _buyer_messages("continue_json")[2]["messages"]
    own, partner = seller_turn2[1], seller_turn2[2]
    assert own["role"] == "assistant" and json.loads(own["content"])["price"] == 200
    assert partner == {"role": "user", "content": "Buyer"}


def test_unknown_message_format_rejected():
    import pytest
    with pytest.raises(ValueError):
        _config("nope")


def _validated_config():
    from harness.config import ProtocolConfig
    return RunConfig(name="t", model=CONFIG.model, protocol=ProtocolConfig(validate_actions=True))


def test_validation_regenerates_opening_accept():
    client = FakeClient([reply("accept", 200), reply("offer", 200), reply("accept", 200)])
    result = asyncio.run(run_negotiation(client, _validated_config(), SCENARIO, "scenario-00",
                                         Condition("en", "none", "neutral"), 0))
    assert [t["action"] for t in result["turns"]] == ["offer", "accept"]
    assert result["calls"][0]["violation"] == "opening_not_offer"
    assert result["calls"][1]["attempt"] == 1


def test_validation_regenerates_priceless_counter_offer():
    client = FakeClient([reply("offer", 200), reply("counter-offer", None), reply("counter-offer", 150), reply("accept", 150)])
    result = asyncio.run(run_negotiation(client, _validated_config(), SCENARIO, "scenario-00",
                                         Condition("en", "none", "neutral"), 0))
    assert result["final_price"] == 150
    assert [c.get("violation") for c in result["calls"]] == [None, "offer_without_price", None, None]


def test_without_validation_opening_accept_ends_the_run():
    result, _ = negotiate([reply("accept", 200)])
    assert result["outcome"] == "agreed" and result["n_turns"] == 1
