"""Run configuration, loaded from YAML.

The config hash is part of every transcript's metadata, so any change to the
model, decoding or design produces a new run directory instead of silently
mixing with (and being resume-skipped against) an older run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    hf_id: str
    revision: str
    chat_template_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 512
    base_seed: int = 42


@dataclass(frozen=True)
class DesignConfig:
    scenarios: str = "data/scenarios.json"
    n_repetitions: int = 3
    max_turns: int = 16
    max_parse_retries: int = 3


@dataclass(frozen=True)
class RuntimeConfig:
    concurrency: int = 64


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: ModelConfig
    generation: GenerationConfig = GenerationConfig()
    design: DesignConfig = DesignConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    def config_hash(self) -> str:
        # Runtime settings (concurrency) don't change outputs in intent, so they're
        # left out: re-running with more workers must still resume the same run.
        identity = {key: value for key, value in asdict(self).items() if key != "runtime"}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: Path) -> RunConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return RunConfig(
        name=raw["name"],
        model=ModelConfig(**raw["model"]),
        generation=GenerationConfig(**raw.get("generation", {})),
        design=DesignConfig(**raw.get("design", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )
