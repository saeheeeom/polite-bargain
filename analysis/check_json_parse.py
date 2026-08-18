"""Check that generated result JSON files parse successfully.

By default this scans:
  - results/transcripts
  - results/judged

Pass one or more folder paths to scan a different set of directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_FOLDERS = (Path("results/transcripts"), Path("results/judged"))


def find_json_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*.json") if path.is_file())


def check_json_file(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except json.JSONDecodeError as error:
        return f"{path}: line {error.lineno}, column {error.colno}: {error.msg}"
    except OSError as error:
        return f"{path}: {error}"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether JSON files in result folders parse successfully."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        type=Path,
        default=DEFAULT_FOLDERS,
        help="Folders to scan recursively. Defaults to results/transcripts and results/judged.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_files: list[Path] = []

    for folder in args.folders:
        if not folder.exists():
            print(f"[missing folder] {folder}")
            return 2
        if not folder.is_dir():
            print(f"[not a folder] {folder}")
            return 2
        json_files.extend(find_json_files(folder))

    failures = [message for path in json_files if (message := check_json_file(path))]

    print(f"Checked {len(json_files)} JSON file(s).")
    if failures:
        print(f"Failed to parse {len(failures)} file(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("All JSON files parsed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
