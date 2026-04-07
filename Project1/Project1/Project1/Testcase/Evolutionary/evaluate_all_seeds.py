#!/usr/bin/env python3
"""
One-click evaluator for all generated seed files in Evolutionary maps.

By default, the script scans map*/seed_balanced* and evaluates generated seeds.
For the canonical file named exactly "seed_balanced", result is written to
mapX/result.txt (same style as heuristic).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


MAP_CONFIG = {
    "map1": {"dataset": "dataset1", "budget": 10},
    "map2": {"dataset": "dataset2", "budget": 14},
    "map3": {"dataset": "dataset2", "budget": 14},
    "map4": {"dataset": "dataset3", "budget": 25},
    "map5": {"dataset": "dataset4", "budget": 25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all generated seed files for Evolutionary maps")
    parser.add_argument(
        "--simulations",
        type=int,
        default=5000,
        help="Evaluator MC simulations (default: 5000)",
    )
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="seed_balanced",
        help="Only evaluate files with this prefix (default: seed_balanced)",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="evaluation_summary.csv",
        help="Summary CSV filename (saved in Evolutionary folder)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for deterministic Evaluator MC runs (default: 3407)",
    )
    return parser.parse_args()


def discover_seed_files(map_dir: Path, prefix: str) -> list[Path]:
    candidates = []
    for path in sorted(map_dir.glob(f"{prefix}*")):
        if not path.is_file():
            continue
        if path.name == "seed":
            continue
        candidates.append(path)
    return candidates


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    evaluator = script_dir.parent / "Evaluator" / "Evaluator.py"

    if not evaluator.exists():
        raise FileNotFoundError(f"Evaluator not found: {evaluator}")
    if args.simulations <= 0:
        raise ValueError("--simulations must be positive")

    print("=" * 78)
    print("Evaluate All Evolutionary Seed Files")
    print("=" * 78)
    print(f"Python: {sys.executable}")
    print(f"Evaluator: {evaluator}")
    print(f"Simulations: {args.simulations}")
    print("=" * 78)

    records = []

    for map_name, cfg in MAP_CONFIG.items():
        map_dir = script_dir / map_name
        dataset = map_dir / cfg["dataset"]
        initial = map_dir / "seed"
        budget = int(cfg["budget"])

        if not map_dir.exists():
            print(f"[{map_name}] skip: map directory not found")
            continue

        seed_files = discover_seed_files(map_dir, args.seed_prefix)
        if not seed_files:
            print(f"[{map_name}] no generated seed files found (prefix={args.seed_prefix})")
            continue

        print(f"\n{'=' * 78}")
        print(f"[{map_name}] {len(seed_files)} file(s) to evaluate")
        print(f"{'=' * 78}")

        for seed_file in seed_files:
            if seed_file.name == "seed_balanced":
                output_file = map_dir / "result.txt"
            else:
                output_file = map_dir / f"result_{seed_file.name}.txt"
            cmd = [
                sys.executable,
                str(evaluator),
                "-n",
                str(dataset),
                "-i",
                str(initial),
                "-b",
                str(seed_file),
                "-k",
                str(budget),
                "-o",
                str(output_file),
                "--simulations",
                str(args.simulations),
                "--seed",
                str(args.seed),
            ]

            start = time.perf_counter()
            result = subprocess.run(cmd, cwd=str(script_dir), text=True, capture_output=True)
            elapsed = time.perf_counter() - start

            score = None
            status = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"

            if result.returncode == 0 and output_file.exists():
                try:
                    score = float(output_file.read_text(encoding="utf-8").strip())
                except ValueError:
                    score = None

            print(
                f"{map_name}/{seed_file.name:<28} -> {status:<12} "
                f"time={elapsed:>8.2f}s score={score if score is not None else 'N/A'}"
            )
            if result.returncode != 0:
                if result.stdout:
                    print("  STDOUT:")
                    print(result.stdout.strip())
                if result.stderr:
                    print("  STDERR:")
                    print(result.stderr.strip())

            records.append(
                {
                    "map": map_name,
                    "seed_file": str(seed_file.relative_to(script_dir)),
                    "budget": budget,
                    "score": "" if score is None else f"{score:.6f}",
                    "elapsed_seconds": f"{elapsed:.4f}",
                    "status": status,
                    "result_file": str(output_file.relative_to(script_dir)),
                }
            )

    summary_csv = script_dir / args.summary_csv
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "map",
                "seed_file",
                "budget",
                "score",
                "elapsed_seconds",
                "status",
                "result_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    if not records:
        print("No seed files evaluated.")
    else:
        # Sort by score descending where available.
        def _key(rec):
            if rec["score"] == "":
                return float("-inf")
            return float(rec["score"])

        ranked = sorted(records, key=_key, reverse=True)
        for rec in ranked:
            score_disp = rec["score"] if rec["score"] else "N/A"
            print(
                f"{rec['map']:<5} {rec['seed_file']:<36} "
                f"score={score_disp:<12} time={float(rec['elapsed_seconds']):>8.2f}s "
                f"status={rec['status']}"
            )

    print(f"CSV saved: {summary_csv.name}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
