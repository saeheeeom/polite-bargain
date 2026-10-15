"""Runs the full design for one config against a running vLLM server.

Usage:
    python -m harness.run configs/qwen3-8b.yaml --base-url http://localhost:8000/v1
    python -m harness.run configs/qwen3-8b.yaml --limit 1   # first scenario only (smoke test)

Output: results/runs/<name>-<config_hash>/
    transcripts/<scenario>_<condition>_rep<k>.json   (paper schema + "calls" + "meta")
    manifest.json                                     (run-level provenance and throughput)

Resume-safe: transcripts already on disk are skipped. Each file is written to a
temp name and renamed, so a killed job never leaves a half-written transcript
that a resume would mistake for a finished one.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import openai
from tqdm import tqdm

from harness.config import RunConfig, load_config
from harness.negotiation import run_negotiation
from src.conditions import all_conditions
from src.negotiation import BASE_PROMPT
from src.translations import PERSONA_PHRASING, URGENCY_PHRASING

ROOT = Path(__file__).resolve().parent.parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) != 0
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prompt_hash() -> str:
    """Hash of every prompt template, so a later prompt edit is detectable in old results."""
    from src.agents import AGENT_RESPONSE_INSTRUCTIONS

    blob = json.dumps([BASE_PROMPT, PERSONA_PHRASING, URGENCY_PHRASING, AGENT_RESPONSE_INSTRUCTIONS], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def server_info(base_url: str) -> dict:
    root = base_url.removesuffix("/v1")
    info = {}
    try:
        info["vllm_version"] = httpx.get(f"{root}/version", timeout=10).json().get("version")
        models = httpx.get(f"{base_url}/models", timeout=10).json()["data"]
        info["served_models"] = [model["id"] for model in models]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        info["error"] = str(exc)
    return info


def write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


async def run(config: RunConfig, base_url: str, limit: int | None) -> Path:
    run_dir = ROOT / "results" / "runs" / f"{config.name}-{config.config_hash()}"
    out_dir = run_dir / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = json.loads((ROOT / config.design.scenarios).read_text(encoding="utf-8"))
    if limit:
        scenarios = scenarios[:limit]

    tasks = []
    for index, scenario in enumerate(scenarios):
        scenario_id = f"scenario-{index:02d}"
        for condition in all_conditions():
            for rep in range(config.design.n_repetitions):
                path = out_dir / f"{scenario_id}_{condition.id}_rep{rep}.json"
                if not path.exists():
                    tasks.append((scenario_id, scenario, condition, rep, path))
    total = len(scenarios) * len(all_conditions()) * config.design.n_repetitions
    print(f"{run_dir.name}: {total - len(tasks)}/{total} already done, {len(tasks)} to run")

    meta = {
        "config_hash": config.config_hash(),
        "prompt_hash": prompt_hash(),
        "git_commit": git_commit(),
        "model": config.model.hf_id,
        "model_revision": config.model.revision,
        **server_info(base_url),
    }

    client = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=600, max_retries=0)
    semaphore = asyncio.Semaphore(config.runtime.concurrency)
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()
    started_at = _now()

    async def one(scenario_id, scenario, condition, rep, path):
        async with semaphore:
            begun = _now()
            try:
                result = await run_negotiation(client, config, scenario, scenario_id, condition, rep)
            except Exception as exc:  # recorded and retried on the next resume, not fatal
                failures.append((path.name, repr(exc)))
                return
            result["meta"] = {**meta, "started_at": begun, "finished_at": _now()}
            write_atomic(path, result)

    with tqdm(total=len(tasks)) as bar:
        async def tracked(*args):
            await one(*args)
            bar.update(1)

        await asyncio.gather(*(tracked(*task) for task in tasks))

    elapsed = time.perf_counter() - started
    write_atomic(run_dir / "manifest.json", {
        "config": json.loads(json.dumps(config, default=lambda o: o.__dict__)),
        **meta,
        "started_at": started_at,
        "finished_at": _now(),
        "wall_clock_s": round(elapsed, 2),
        "negotiations_this_session": len(tasks) - len(failures),
        "failures": failures,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    print(f"Done in {elapsed:.1f}s; {len(failures)} failed (rerun to retry). -> {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--limit", type=int, default=None, help="only the first N scenarios")
    args = parser.parse_args()
    asyncio.run(run(load_config(args.config), args.base_url, args.limit))


if __name__ == "__main__":
    main()
