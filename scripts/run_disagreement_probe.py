from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the disagreement-selected probe and persist its leading JSON payload."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("probe_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = Path(__file__).with_name("probe_muon_disagreement_tasks.py")
    command = [sys.executable, str(probe), *args.probe_args]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    start = completed.stdout.find("{")
    if start < 0:
        raise RuntimeError("probe output did not contain a JSON payload")

    payload, _ = json.JSONDecoder().raw_decode(completed.stdout[start:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
