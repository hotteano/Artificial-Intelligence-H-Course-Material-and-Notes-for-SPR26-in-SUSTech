#!/usr/bin/env python3
"""
Unified end-to-end testcase runner.

Runs methods in order: Evaluator -> Heuristic -> Evolutionary.
For each case, it performs evaluation and records:
- solve time (if method has a solver)
- evaluator time
- final evaluator score

Important: this script only passes REQUIRED CLI arguments to each component.
Optional algorithm-impacting parameters are intentionally not provided, so each
component uses its own default settings.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Evaluator/Heuristic/Evolutionary sequentially and write one summary CSV"
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="all_methods_summary.csv",
        help="Output CSV filename saved under Testcase (default: all_methods_summary.csv)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a case fails",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, float, str, str]:
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    return result.returncode, elapsed, result.stdout, result.stderr


def read_score(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent

    evaluator_script = root / "Evaluator" / "Evaluator.py"
    heuristic_script = root / "Heuristic" / "IEMP_Heur.py"
    evolutionary_script = root / "Evolutionary" / "IEMP_Evol.py"

    pipelines = [
        {
            "method": "Evaluator",
            "solver": None,
            "cases": [
                {
                    "map": "map1",
                    "network": "Evaluator/map1/dataset1",
                    "initial": "Evaluator/map1/seed",
                    "balanced": "Evaluator/map1/seed_balanced",
                    "budget": 10,
                    "result": "Evaluator/map1/result.txt",
                },
                {
                    "map": "map2",
                    "network": "Evaluator/map2/dataset2",
                    "initial": "Evaluator/map2/seed",
                    "balanced": "Evaluator/map2/seed_balanced",
                    "budget": 15,
                    "result": "Evaluator/map2/result.txt",
                },
            ],
        },
        {
            "method": "Heuristic",
            "solver": heuristic_script,
            "cases": [
                {
                    "map": "map1",
                    "network": "Heuristic/map1/dataset1",
                    "initial": "Heuristic/map1/seed",
                    "balanced": "Heuristic/map1/seed_balanced",
                    "budget": 10,
                    "result": "Heuristic/map1/result.txt",
                },
                {
                    "map": "map2",
                    "network": "Heuristic/map2/dataset2",
                    "initial": "Heuristic/map2/seed",
                    "balanced": "Heuristic/map2/seed_balanced",
                    "budget": 15,
                    "result": "Heuristic/map2/result.txt",
                },
                {
                    "map": "map3",
                    "network": "Heuristic/map3/dataset2",
                    "initial": "Heuristic/map3/seed2",
                    "balanced": "Heuristic/map3/seed_balanced",
                    "budget": 15,
                    "result": "Heuristic/map3/result.txt",
                },
                {
                    "map": "map4",
                    "network": "Heuristic/map4/dataset3",
                    "initial": "Heuristic/map4/seed",
                    "balanced": "Heuristic/map4/seed_balanced",
                    "budget": 25,
                    "result": "Heuristic/map4/result.txt",
                },
                {
                    "map": "map5",
                    "network": "Heuristic/map5/dataset4",
                    "initial": "Heuristic/map5/seed",
                    "balanced": "Heuristic/map5/seed_balanced",
                    "budget": 25,
                    "result": "Heuristic/map5/result.txt",
                },
            ],
        },
        {
            "method": "Evolutionary",
            "solver": evolutionary_script,
            "cases": [
                {
                    "map": "map1",
                    "network": "Evolutionary/map1/dataset1",
                    "initial": "Evolutionary/map1/seed",
                    "balanced": "Evolutionary/map1/seed_balanced",
                    "budget": 10,
                    "result": "Evolutionary/map1/result.txt",
                },
                {
                    "map": "map2",
                    "network": "Evolutionary/map2/dataset2",
                    "initial": "Evolutionary/map2/seed",
                    "balanced": "Evolutionary/map2/seed_balanced",
                    "budget": 14,
                    "result": "Evolutionary/map2/result.txt",
                },
                {
                    "map": "map3",
                    "network": "Evolutionary/map3/dataset2",
                    "initial": "Evolutionary/map3/seed",
                    "balanced": "Evolutionary/map3/seed_balanced",
                    "budget": 14,
                    "result": "Evolutionary/map3/result.txt",
                },
                {
                    "map": "map4",
                    "network": "Evolutionary/map4/dataset3",
                    "initial": "Evolutionary/map4/seed",
                    "balanced": "Evolutionary/map4/seed_balanced",
                    "budget": 25,
                    "result": "Evolutionary/map4/result.txt",
                },
                {
                    "map": "map5",
                    "network": "Evolutionary/map5/dataset4",
                    "initial": "Evolutionary/map5/seed",
                    "balanced": "Evolutionary/map5/seed_balanced",
                    "budget": 25,
                    "result": "Evolutionary/map5/result.txt",
                },
                {
                    "map": "map6",
                    "network": "Evolutionary/map6/dataset6_hard",
                    "initial": "Evolutionary/map6/seed",
                    "balanced": "Evolutionary/map6/seed_balanced",
                    "budget": 25,
                    "result": "Evolutionary/map6/result.txt",
                },
                {
                    "map": "map7",
                    "network": "Evolutionary/map7/dataset7_hard",
                    "initial": "Evolutionary/map7/seed",
                    "balanced": "Evolutionary/map7/seed_balanced",
                    "budget": 25,
                    "result": "Evolutionary/map7/result.txt",
                },
            ],
        },
    ]

    print("=" * 84)
    print("Unified Test Runner: Evaluator -> Heuristic -> Evolutionary")
    print("Only required CLI arguments are used.")
    print("=" * 84)

    rows: list[dict[str, str]] = []

    for pipe in pipelines:
        method = pipe["method"]
        solver = pipe["solver"]
        cases = pipe["cases"]

        print(f"\n{'=' * 84}")
        print(f"[{method}] start ({len(cases)} case(s))")
        print(f"{'=' * 84}")

        for case in cases:
            map_name = case["map"]
            network = root / case["network"]
            initial = root / case["initial"]
            balanced = root / case["balanced"]
            budget = int(case["budget"])
            result_path = root / case["result"]

            print(f"\n[{method}/{map_name}] running...")

            solve_elapsed = 0.0
            solve_status = "SKIPPED"

            if solver is not None:
                solve_cmd = [
                    sys.executable,
                    str(solver),
                    "-n",
                    str(network),
                    "-i",
                    str(initial),
                    "-b",
                    str(balanced),
                    "-k",
                    str(budget),
                ]
                rc, solve_elapsed, out, err = run_cmd(solve_cmd, cwd=root)
                solve_status = "OK" if rc == 0 else f"FAILED({rc})"
                if rc != 0:
                    if out.strip():
                        print("SOLVER STDOUT:")
                        print(out.strip())
                    if err.strip():
                        print("SOLVER STDERR:")
                        print(err.strip())

            eval_elapsed = 0.0
            eval_status = "NOT_RUN"
            score = None

            if solve_status == "OK" or method == "Evaluator":
                eval_cmd = [
                    sys.executable,
                    str(evaluator_script),
                    "-n",
                    str(network),
                    "-i",
                    str(initial),
                    "-b",
                    str(balanced),
                    "-k",
                    str(budget),
                    "-o",
                    str(result_path),
                ]
                rc, eval_elapsed, out, err = run_cmd(eval_cmd, cwd=root)
                eval_status = "OK" if rc == 0 else f"FAILED({rc})"
                if rc == 0:
                    score = read_score(result_path)
                else:
                    if out.strip():
                        print("EVAL STDOUT:")
                        print(out.strip())
                    if err.strip():
                        print("EVAL STDERR:")
                        print(err.strip())

            overall_status = "OK" if (solve_status in {"OK", "SKIPPED"} and eval_status == "OK") else "FAILED"
            total_elapsed = solve_elapsed + eval_elapsed

            print(
                f"[{method}/{map_name}] status={overall_status} "
                f"solve={solve_elapsed:.4f}s eval={eval_elapsed:.4f}s "
                f"score={score if score is not None else 'N/A'}"
            )

            rows.append(
                {
                    "method": method,
                    "map": map_name,
                    "budget": str(budget),
                    "solve_seconds": f"{solve_elapsed:.4f}",
                    "evaluate_seconds": f"{eval_elapsed:.4f}",
                    "total_seconds": f"{total_elapsed:.4f}",
                    "score": "" if score is None else f"{score:.6f}",
                    "solve_status": solve_status,
                    "evaluate_status": eval_status,
                    "overall_status": overall_status,
                    "network_file": str(case["network"]),
                    "initial_seed_file": str(case["initial"]),
                    "balanced_seed_file": str(case["balanced"]),
                    "result_file": str(case["result"]),
                }
            )

            if args.stop_on_error and overall_status != "OK":
                print("Stop on first error as requested by --stop-on-error.")
                summary_path = root / args.summary_csv
                with open(summary_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "method",
                            "map",
                            "budget",
                            "solve_seconds",
                            "evaluate_seconds",
                            "total_seconds",
                            "score",
                            "solve_status",
                            "evaluate_status",
                            "overall_status",
                            "network_file",
                            "initial_seed_file",
                            "balanced_seed_file",
                            "result_file",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Partial CSV saved: {summary_path.name}")
                return 1

    summary_path = root / args.summary_csv
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "map",
                "budget",
                "solve_seconds",
                "evaluate_seconds",
                "total_seconds",
                "score",
                "solve_status",
                "evaluate_status",
                "overall_status",
                "network_file",
                "initial_seed_file",
                "balanced_seed_file",
                "result_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 84}")
    print("ALL DONE")
    print(f"CSV saved: {summary_path.name}")
    print(f"Rows: {len(rows)}")
    print(f"{'=' * 84}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
