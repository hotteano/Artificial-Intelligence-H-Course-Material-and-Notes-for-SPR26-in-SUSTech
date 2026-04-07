#!/usr/bin/env python3
"""
Run Evaluator on map1/map2 with fixed simulation counts: 5000, 10000, 20000.

Outputs:
- Per-run result files under map1/map2, e.g. timed_result_5000.txt
- A CSV summary file in this folder
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional


EVALUATOR_FILE = "Evaluator.py"
DEFAULT_SIM_LEVELS = [5000, 10000, 20000]
TEST_CASES = [
    {
        "name": "map1",
        "network": "map1/dataset1",
        "initial": "map1/seed",
        "balanced": "map1/seed_balanced",
        "budget": 10,
        "output": "map1/timed_result.txt",
    },
    {
        "name": "map2",
        "network": "map2/dataset2",
        "initial": "map2/seed",
        "balanced": "map2/seed_balanced",
        "budget": 15,
        "output": "map2/timed_result.txt",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run map1/map2 with simulation counts 5000/10000/20000"
    )
    parser.add_argument(
        "--sim-levels",
        type=int,
        nargs="+",
        default=DEFAULT_SIM_LEVELS,
        help="Simulation levels, default: 5000 10000 20000",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="timing_score_5k_10k_20k.csv",
        help="Summary CSV filename",
    )
    parser.add_argument(
        "--print-evaluator-output",
        action="store_true",
        help="Print full Evaluator stdout/stderr",
    )
    return parser.parse_args()


def output_path_for_stage(base_output: str, simulations: int) -> str:
    output_path = Path(base_output)
    return str(output_path.with_name(f"{output_path.stem}_{simulations}{output_path.suffix}"))


def run_one_case(
    test_case: Dict[str, object],
    simulations: int,
    script_dir: Path,
    print_output: bool,
) -> Dict[str, object]:
    output_file = output_path_for_stage(str(test_case["output"]), simulations)
    cmd = [
        sys.executable,
        EVALUATOR_FILE,
        "-n",
        str(test_case["network"]),
        "-i",
        str(test_case["initial"]),
        "-b",
        str(test_case["balanced"]),
        "-k",
        str(test_case["budget"]),
        "-o",
        output_file,
        "--simulations",
        str(simulations),
    ]

    print(f"\nRunning {test_case['name']} @ simulations={simulations}")
    print("Command:", " ".join(cmd))

    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(script_dir), capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if print_output:
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

    score: Optional[float] = None
    status = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"

    if result.returncode == 0:
        score_path = script_dir / output_file
        if score_path.exists():
            try:
                score = float(score_path.read_text(encoding="utf-8").strip())
            except ValueError:
                score = None
    else:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    return {
        "case": str(test_case["name"]),
        "simulations": simulations,
        "elapsed_seconds": elapsed,
        "score": score,
        "status": status,
        "output_file": output_file,
    }


def write_summary_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "simulations",
                "elapsed_seconds",
                "score",
                "status",
                "output_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "simulations": row["simulations"],
                    "elapsed_seconds": f"{float(row['elapsed_seconds']):.6f}",
                    "score": "" if row["score"] is None else f"{float(row['score']):.6f}",
                    "status": row["status"],
                    "output_file": row["output_file"],
                }
            )


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    sim_levels = sorted(set(args.sim_levels))
    if any(level <= 0 for level in sim_levels):
        raise ValueError("All --sim-levels values must be positive")

    print("=" * 70)
    print("Run Evaluator: map1/map2 with staged simulation levels")
    print("Simulation levels:", sim_levels)
    print("=" * 70)

    rows: List[Dict[str, object]] = []

    for simulations in sim_levels:
        for test_case in TEST_CASES:
            row = run_one_case(
                test_case=test_case,
                simulations=simulations,
                script_dir=script_dir,
                print_output=args.print_evaluator_output,
            )
            rows.append(row)

            score_text = "N/A" if row["score"] is None else f"{float(row['score']):.6f}"
            print(
                f"[{row['case']}] sims={row['simulations']} "
                f"time={float(row['elapsed_seconds']):.6f}s "
                f"score={score_text} status={row['status']}"
            )

    summary_csv_path = script_dir / args.summary_csv
    write_summary_csv(summary_csv_path, rows)

    print("\n" + "=" * 70)
    print("Summary CSV saved:", summary_csv_path.name)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())