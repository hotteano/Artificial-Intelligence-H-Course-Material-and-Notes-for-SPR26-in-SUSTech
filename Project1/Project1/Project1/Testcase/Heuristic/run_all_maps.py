#!/usr/bin/env python3
"""
Run Heuristic Algorithm on All Maps (map1-map5)
Usage:
    python run_all_maps.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import time
import sys

TEST_CASES = [
    ("map1", "map1/dataset1", "map1/seed", "map1/seed_balanced", 10),
    ("map2", "map2/dataset2", "map2/seed", "map2/seed_balanced", 15),
    ("map3", "map3/dataset2", "map3/seed2", "map3/seed_balanced", 15),
    ("map4", "map4/dataset3", "map4/seed", "map4/seed_balanced", 15),
    ("map5", "map5/dataset4", "map5/seed", "map5/seed_balanced", 15),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heuristic solver on all maps")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="timing_summary.csv",
        help="Summary CSV filename (saved in Heuristic folder)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print("Heuristic Algorithm - All Maps Test")
    print("=" * 70)
    print(f"Total test cases: {len(TEST_CASES)}")
    print("=" * 70)

    records = []

    for name, network, initial, output, budget in TEST_CASES:
        print(f"\n{'=' * 70}")
        print(f"[{name}] Starting... (budget={budget})")
        print(f"{'=' * 70}")

        cmd = [
            sys.executable,
            str(script_dir / "IEMP_Heur.py"),
            "-n",
            network,
            "-i",
            initial,
            "-b",
            output,
            "-k",
            str(budget),
        ]

        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=str(script_dir), capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        status = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print(
            f"\n[{name}] {status} in {elapsed:.2f} seconds "
            f"({elapsed / 60:.2f} minutes)"
        )

        records.append(
            {
                "map": name,
                "budget": budget,
                "output_seed": output,
                "elapsed_seconds": f"{elapsed:.4f}",
                "status": status,
            }
        )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Map':<10} {'Budget':<10} {'Time (s)':<15} {'Time (min)':<12} {'Status'}")
    print("-" * 70)

    total_time = 0.0
    for rec in records:
        elapsed = float(rec["elapsed_seconds"])
        total_time += elapsed
        print(
            f"{rec['map']:<10} {rec['budget']:<10} {elapsed:<15.2f} "
            f"{elapsed / 60:<12.2f} {rec['status']}"
        )

    summary_csv = script_dir / args.summary_csv
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["map", "budget", "output_seed", "elapsed_seconds", "status"],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"{'=' * 70}")
    print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"CSV saved: {summary_csv.name}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
