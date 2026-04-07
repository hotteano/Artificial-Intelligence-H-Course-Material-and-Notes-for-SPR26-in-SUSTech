#!/usr/bin/env python3
"""
One-click timed runner for IEMP_Evol.py on all Evolutionary maps.

Usage:
    python run_all_maps_timed.py

Optional parameters allow overriding EA hyperparameters in batch runs.
Output seed file is fixed to mapX/seed_balanced (same naming as heuristic).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


TEST_CASES = [
    # (map_name, dataset_filename, budget)
    # case0-case2 are existing; case3-case4 are added per requirement.
    ("map1", "dataset1", 10),
    ("map2", "dataset2", 14),
    ("map3", "dataset2", 14),
    ("map4", "dataset3", 15),
    ("map5", "dataset4", 15),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Timed batch runner for IEMP_Evol.py")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--mc-coarse", type=int, default=60)
    parser.add_argument("--mc-fine", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed; if omitted, each run is stochastic",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="timing_summary.csv",
        help="Summary CSV filename (saved in Evolutionary folder)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    solver_path = script_dir / "IEMP_Evol.py"

    if not solver_path.exists():
        raise FileNotFoundError(f"Solver not found: {solver_path}")

    print("=" * 72)
    print("IEMP Evolutionary - Timed Batch Run")
    print("=" * 72)
    print(f"Python: {sys.executable}")
    print(f"Solver: {solver_path.name}")
    print(f"Cases: {len(TEST_CASES)}")
    print("=" * 72)

    records = []

    for map_name, dataset_name, budget in TEST_CASES:
        map_dir = script_dir / map_name
        dataset = map_dir / dataset_name
        initial = map_dir / "seed"
        balanced = map_dir / "seed_balanced"

        cmd = [
            sys.executable,
            str(solver_path),
            "-n",
            str(dataset),
            "-i",
            str(initial),
            "-b",
            str(balanced),
            "-k",
            str(budget),
            "--pop-size",
            str(args.pop_size),
            "--generations",
            str(args.generations),
            "--crossover-rate",
            str(args.crossover_rate),
            "--mutation-rate",
            str(args.mutation_rate),
            "--elitism",
            str(args.elitism),
            "--mc-coarse",
            str(args.mc_coarse),
            "--mc-fine",
            str(args.mc_fine),
        ]
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])

        print(f"\n{'=' * 72}")
        print(f"[{map_name}] Running... budget={budget}")
        print(f"Output: {balanced.relative_to(script_dir)}")
        print(f"{'=' * 72}")

        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=str(script_dir), text=True)
        elapsed = time.perf_counter() - start

        status = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"
        print(f"[{map_name}] {status}, elapsed={elapsed:.2f}s ({elapsed/60:.2f} min)")

        records.append(
            {
                "map": map_name,
                "budget": budget,
                "output_seed": str(balanced.relative_to(script_dir)),
                "elapsed_seconds": f"{elapsed:.4f}",
                "status": status,
            }
        )

    summary_csv = script_dir / args.summary_csv
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["map", "budget", "output_seed", "elapsed_seconds", "status"],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    total = 0.0
    for rec in records:
        sec = float(rec["elapsed_seconds"])
        total += sec
        print(
            f"{rec['map']:<8} budget={rec['budget']:<2} "
            f"time={sec:>8.2f}s status={rec['status']:<12} output={rec['output_seed']}"
        )
    print(f"Total elapsed: {total:.2f}s ({total/60:.2f} min)")
    print(f"CSV saved: {summary_csv.name}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
