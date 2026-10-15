"""The 2 (language) x 2 (seller urgency) x 3 (buyer persona) design, as the analysis sees it."""
import itertools

LANGUAGES = ["en", "ko"]
URGENCY_LEVELS = ["none", "strong"]
PERSONAS = ["cooperative", "neutral", "headstrong"]
CONDITION_COLUMNS = ["language", "urgency", "persona"]
# Order matters: contrast weight vectors below are indexed by position in this list.
CONDITIONS = list(itertools.product(LANGUAGES, URGENCY_LEVELS, PERSONAS))

ROLES = ["buyer", "seller"]
MAX_TURNS = 16
N_REPETITIONS = 3

PRICE_ACTIONS = {"offer", "counter-offer"}
TERMINAL_ACTIONS = {"accept", "reject"}
ALLOWED_ACTIONS = PRICE_ACTIONS | TERMINAL_ACTIONS
VALID_OUTCOMES = {"agreed", "abandoned", "max_turns_reached"}
