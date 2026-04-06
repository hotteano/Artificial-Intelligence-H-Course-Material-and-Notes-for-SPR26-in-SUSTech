#!/usr/bin/env python3
"""
Evaluate All Heuristic Results using Evaluator
Calls Evaluator.py on all generated seed_balanced files and measures timing
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import time
import sys
import os

# Evaluator path (relative to Heuristic directory)
EVALUATOR_PATH = "../Evaluator/Evaluator.py"

# Test cases configuration
# Format: (name, network, initial, balanced, budget, output_result)
TEST_CASES = [
    ("map1", "../Evaluator/map1/dataset1", "../Evaluator/map1/seed", "map1/seed_balanced", 10, "map1/result.txt"),
    ("map2", "../Evaluator/map2/dataset2", "../Evaluator/map2/seed", "map2/seed_balanced", 15, "map2/result.txt"),
    ("map3", "map3/dataset2", "map3/seed2", "map3/seed_balanced", 15, "map3/result.txt"),
    ("map4", "map4/dataset3", "map4/seed", "map4/seed_balanced", 15, "map4/result.txt"),
    ("map5", "map5/dataset4", "map5/seed", "map5/seed_balanced", 15, "map5/result.txt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all heuristic outputs")
    parser.add_argument(
        "--simulations",
        type=int,
        default=60000,
        help="Number of MC simulations for evaluator (default: 60000)",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="evaluation_summary.csv",
        help="Summary CSV filename (saved in Heuristic folder)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print("Evaluate All Heuristic Results")
    print("=" * 70)
    print(f"Evaluator: {EVALUATOR_PATH}")
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Note: Using MC={args.simulations} simulations")
    print("=" * 70)

    records = []

    for name, network, initial, balanced, budget, output in TEST_CASES:
        print(f"\n{'=' * 70}")
        print(f"[{name}] Evaluating...")
        print(f"  Network: {network}")
        print(f"  Initial: {initial}")
        print(f"  Balanced: {balanced}")
        print(f"  Budget: {budget}")
        print(f"{'=' * 70}")

        if not os.path.exists(balanced):
            print(f"[ERROR] Balanced seed file not found: {balanced}")
            print(f"[SKIP] Run heuristic algorithm first to generate {balanced}")
            records.append(
                {
                    "map": name,
                    "seed_file": balanced,
                    "budget": budget,
                    "score": "",
                    "elapsed_seconds": "",
                    "status": "FILE NOT FOUND",
                    "result_file": output,
                }
            )
            continue

        cmd = [
            sys.executable,
            EVALUATOR_PATH,
            "-n",
            network,
            "-i",
            initial,
            "-b",
            balanced,
            "-k",
            str(budget),
            "-o",
            output,
            "--simulations",
            str(args.simulations),
        ]

        start = time.perf_counter()
        try:
            result = subprocess.run(cmd, cwd=str(script_dir), capture_output=True, text=True, check=True)
            elapsed = time.perf_counter() - start

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            score = None
            try:
                with open(output, "r", encoding="utf-8") as f:
                    score = float(f.read().strip())
            except (ValueError, OSError):
                score = None

            print(f"\n[{name}] EVALUATION COMPLETED")
            print(f"  Time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
            if score is not None:
                print(f"  Score: {score:.4f}")
            print(f"  Result saved to: {output}")

            records.append(
                {
                    "map": name,
                    "seed_file": balanced,
                    "budget": budget,
                    "score": "" if score is None else f"{score:.6f}",
                    "elapsed_seconds": f"{elapsed:.4f}",
                    "status": "OK",
                    "result_file": output,
                }
            )

        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start
            print(f"[ERROR] Evaluation failed for {name}")
            print(f"  STDOUT: {e.stdout}")
            print(f"  STDERR: {e.stderr}")
            records.append(
                {
                    "map": name,
                    "seed_file": balanced,
                    "budget": budget,
                    "score": "",
                    "elapsed_seconds": f"{elapsed:.4f}",
                    "status": "FAILED",
                    "result_file": output,
                }
            )

    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Map':<10} {'Budget':<10} {'Time (s)':<12} {'Time (min)':<12} {'Score':<15} {'Status'}")
    print("-" * 70)

    total_time = 0.0
    for rec in records:
        if rec["elapsed_seconds"]:
            elapsed = float(rec["elapsed_seconds"])
            total_time += elapsed
            time_str = f"{elapsed:<12.2f}"
            min_str = f"{elapsed / 60:<12.2f}"
        else:
            time_str = "N/A         "
            min_str = "N/A         "

        score_str = f"{float(rec['score']):<15.4f}" if rec["score"] else "N/A             "
        print(f"{rec['map']:<10} {rec['budget']:<10} {time_str} {min_str} {score_str} {rec['status']}")

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

    print(f"{'=' * 70}")
    print(f"Total evaluation time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"CSV saved: {summary_csv.name}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
