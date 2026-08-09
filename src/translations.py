"""
EN/KO text for the urgency and persona manipulations.

These get inserted into the seller's (urgency) or buyer's (persona) system prompt,
in whichever language that negotiation's `language` condition specifies.

IMPORTANT: the Korean strings below are a first machine-assisted draft, not yet
verified. Per the lightweight verification plan in the research statement, do a
read-through pass over ALL of these (there are only 5 short strings total) before
running the real experiment. Search this file for "TODO: verify Korean".
"""

URGENCY_PHRASING = {
    "none": {
        "en": "",
        "ko": "",
    },
    "strong": {
        "en": "You need the cash quickly.",
        "ko": "당신은 급하게 현금이 필요합니다.",  # TODO: verify Korean
    },
}

PERSONA_PHRASING = {
    "cooperative": {
        "en": (
            "You prioritize reaching a deal that works for both sides quickly. "
            "You are willing to make concessions readily rather than prolong the negotiation."
        ),
        "ko": (
            "당신은 양쪽 모두에게 괜찮은 합의를 빠르게 이루는 것을 중요하게 생각합니다. "
            "협상을 오래 끌기보다는 기꺼이 양보하는 편입니다."
        ),  # TODO: verify Korean
    },
    "neutral": {
        "en": "",  # baseline — no extra persona instruction
        "ko": "",
    },
    "headstrong": {
        "en": (
            "You anchor firmly on your target price. You concede slowly and reluctantly, "
            "and only when it seems necessary to avoid losing the deal entirely."
        ),
        "ko": (
            "당신은 목표 가격을 확고하게 고수합니다. 거래가 완전히 무산될 것 같을 때만, "
            "그것도 천천히 마지못해 양보합니다."
        ),  # TODO: verify Korean
    },
}


def get_urgency_text(urgency: str, language: str) -> str:
    return URGENCY_PHRASING[urgency][language]


def get_persona_text(persona: str, language: str) -> str:
    return PERSONA_PHRASING[persona][language]
