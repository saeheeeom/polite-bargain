"""
Defines the 2 (language) x 2 (seller urgency) x 3 (buyer persona) = 12 condition grid.

Each Condition is one cell of the factorial design. `run_experiment.py` crosses this
with the sampled scenarios and repetition count to produce the full call plan.
"""
from dataclasses import dataclass
from itertools import product

LANGUAGES = ["en", "ko"]
URGENCY_LEVELS = ["none", "strong"]
PERSONAS = ["cooperative", "neutral", "headstrong"]

N_REPETITIONS = 3
N_SCENARIOS = 12  # set by data_loader.py sampling; kept here too as the single source of truth


@dataclass(frozen=True)
class Condition:
    language: str      # "en" | "ko"
    urgency: str        # "none" | "strong"
    persona: str         # "cooperative" | "neutral" | "headstrong"

    @property
    def id(self) -> str:
        return f"{self.language}-{self.urgency}-{self.persona}"


def all_conditions() -> list[Condition]:
    """Returns all 12 conditions in the factorial design."""
    return [
        Condition(language=lang, urgency=urg, persona=per)
        for lang, urg, per in product(LANGUAGES, URGENCY_LEVELS, PERSONAS)
    ]


def total_negotiations(n_scenarios: int = N_SCENARIOS, n_repetitions: int = N_REPETITIONS) -> int:
    return len(all_conditions()) * n_scenarios * n_repetitions


if __name__ == "__main__":
    conds = all_conditions()
    print(f"{len(conds)} conditions:")
    for c in conds:
        print(f"  {c.id}")
    print(f"\nTotal negotiations at {N_SCENARIOS} scenarios x {N_REPETITIONS} reps: "
          f"{total_negotiations()}")
