#!/usr/bin/env python3
"""
Evaluator Stage Timing Script

Runs staged experiments with multiple Monte Carlo simulation counts,
records execution time and final score, and exports CSV reports.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import time
import os
import sys
from typing import Dict, List, Optional

# Configuration
EVALUATOR_PATH = "Evaluator.py"
DEFAULT_SIM_LEVELS = [30000, 40000, 50000, 60000]
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
        description="Run staged Evaluator timing/score experiments"
    )
    parser.add_argument(
        "--sim-levels",
        type=int,
        nargs="+",
        default=DEFAULT_SIM_LEVELS,
        help="Simulation stages, e.g. --sim-levels 30000 40000 50000 60000",
    )
    parser.add_argument(
        "--time-threshold",
        type=float,
        default=48.0,
        help="Pass/fail threshold (seconds) for timing report",
    )
    parser.add_argument(
        "--detail-csv",
        type=str,
        default="timing_score_stages.csv",
        help="Detailed CSV output filename",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="timing_score_stage_summary.csv",
        help="Stage summary CSV output filename",
    )
    parser.add_argument(
        "--print-evaluator-output",
        action="store_true",
        help="Print full Evaluator stdout/stderr for each run",
    )
    return parser.parse_args()


def _output_path_for_stage(base_output: str, simulations: int) -> str:
    output_path = Path(base_output)
    return str(output_path.with_name(f"{output_path.stem}_{simulations}{output_path.suffix}"))


def run_evaluator(
    test_case: Dict[str, object],
    simulations: int,
    script_dir: Path,
    print_output: bool,
) -> Dict[str, object]:
    """Run evaluator on one test case and one simulation level."""
    output_file = _output_path_for_stage(str(test_case["output"]), simulations)

    cmd = [
        sys.executable,
        EVALUATOR_PATH,
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

    print(f"\n{'=' * 70}")
    print(f"Running: {test_case['name']} (simulations={simulations})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    start_time = time.perf_counter()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(script_dir),
    )
    elapsed = time.perf_counter() - start_time

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
        print(f"Error running evaluator for {test_case['name']} @ {simulations}: {status}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

    return {
        "case": str(test_case["name"]),
        "simulations": simulations,
        "budget": int(test_case["budget"]),
        "network": str(test_case["network"]),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "score": score,
        "status": status,
        "output_file": output_file,
    }


def write_detail_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "simulations",
                "budget",
                "network",
                "elapsed_seconds",
                "elapsed_minutes",
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
                    "budget": row["budget"],
                    "network": row["network"],
                    "elapsed_seconds": f"{float(row['elapsed_seconds']):.6f}",
                    "elapsed_minutes": f"{float(row['elapsed_minutes']):.6f}",
                    "score": "" if row["score"] is None else f"{float(row['score']):.6f}",
                    "status": row["status"],
                    "output_file": row["output_file"],
                }
            )


def build_stage_summary(rows: List[Dict[str, object]], sim_levels: List[int]) -> List[Dict[str, object]]:
    by_case: Dict[str, Dict[int, Dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), {})[int(row["simulations"])] = row

    summary_rows: List[Dict[str, object]] = []

    for idx, sim in enumerate(sim_levels):
        stage_rows = [r for r in rows if int(r["simulations"]) == sim and str(r["status"]) == "OK"]
        if stage_rows:
            avg_time = sum(float(r["elapsed_seconds"]) for r in stage_rows) / len(stage_rows)
            valid_scores = [float(r["score"]) for r in stage_rows if r["score"] is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        else:
            avg_time = None
            avg_score = None

        mean_abs_delta = None
        max_abs_delta = None
        avg_time_ratio = None

        if idx > 0:
            prev_sim = sim_levels[idx - 1]
            score_deltas = []
            time_ratios = []
            for case_name, case_data in by_case.items():
                if sim in case_data and prev_sim in case_data:
                    cur = case_data[sim]
                    prev = case_data[prev_sim]
                    if cur["score"] is not None and prev["score"] is not None:
                        score_deltas.append(abs(float(cur["score"]) - float(prev["score"])))
                    prev_time = float(prev["elapsed_seconds"])
                    cur_time = float(cur["elapsed_seconds"])
                    if prev_time > 0:
                        time_ratios.append(cur_time / prev_time)
            if score_deltas:
                mean_abs_delta = sum(score_deltas) / len(score_deltas)
                max_abs_delta = max(score_deltas)
            if time_ratios:
                avg_time_ratio = sum(time_ratios) / len(time_ratios)

        summary_rows.append(
            {
                "simulations": sim,
                "ok_cases": len(stage_rows),
                "avg_elapsed_seconds": avg_time,
                "avg_score": avg_score,
                "mean_abs_score_delta_vs_prev": mean_abs_delta,
                "max_abs_score_delta_vs_prev": max_abs_delta,
                "avg_time_ratio_vs_prev": avg_time_ratio,
            }
        )

    return summary_rows


def write_stage_summary_csv(csv_path: Path, summary_rows: List[Dict[str, object]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "simulations",
                "ok_cases",
                "avg_elapsed_seconds",
                "avg_score",
                "mean_abs_score_delta_vs_prev",
                "max_abs_score_delta_vs_prev",
                "avg_time_ratio_vs_prev",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "simulations": row["simulations"],
                    "ok_cases": row["ok_cases"],
                    "avg_elapsed_seconds": ""
                    if row["avg_elapsed_seconds"] is None
                    else f"{float(row['avg_elapsed_seconds']):.6f}",
                    "avg_score": ""
                    if row["avg_score"] is None
                    else f"{float(row['avg_score']):.6f}",
                    "mean_abs_score_delta_vs_prev": ""
                    if row["mean_abs_score_delta_vs_prev"] is None
                    else f"{float(row['mean_abs_score_delta_vs_prev']):.6f}",
                    "max_abs_score_delta_vs_prev": ""
                    if row["max_abs_score_delta_vs_prev"] is None
                    else f"{float(row['max_abs_score_delta_vs_prev']):.6f}",
                    "avg_time_ratio_vs_prev": ""
                    if row["avg_time_ratio_vs_prev"] is None
                    else f"{float(row['avg_time_ratio_vs_prev']):.6f}",
                }
            )


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    sim_levels = sorted(set(args.sim_levels))
    if any(x <= 0 for x in sim_levels):
        raise ValueError("All --sim-levels values must be positive integers")

    print("=" * 70)
    print("Evaluator Stage Timing Test")
    print("=" * 70)
    print(f"Python: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Simulation stages: {sim_levels}")
    print(f"Threshold: {args.time_threshold:.2f}s")

    detail_rows: List[Dict[str, object]] = []

    for sim in sim_levels:
        print(f"\n{'=' * 70}")
        print(f"STAGE: simulations = {sim}")
        print(f"{'=' * 70}")
        for test_case in TEST_CASES:
            row = run_evaluator(
                test_case=test_case,
                simulations=sim,
                script_dir=script_dir,
                print_output=args.print_evaluator_output,
            )
            detail_rows.append(row)

            status_48 = "PASS" if float(row["elapsed_seconds"]) < args.time_threshold else "FAIL"
            score_display = "N/A" if row["score"] is None else f"{float(row['score']):.6f}"
            print(
                f"[{row['case']}] time={float(row['elapsed_seconds']):.6f}s "
                f"score={score_display} run_status={row['status']} threshold={status_48}"
            )

    print(f"\n{'=' * 70}")
    print("DETAILED SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Case':<10} {'Sims':<10} {'Time (s)':<14} {'Score':<14} {'<Threshold':<12} {'Status'}"
    )
    print("-" * 70)
    for row in detail_rows:
        under_threshold = "PASS" if float(row["elapsed_seconds"]) < args.time_threshold else "FAIL"
        score_display = "N/A" if row["score"] is None else f"{float(row['score']):.6f}"
        print(
            f"{row['case']:<10} {int(row['simulations']):<10} "
            f"{float(row['elapsed_seconds']):<14.6f} {score_display:<14} {under_threshold:<12} {row['status']}"
        )

    summary_rows = build_stage_summary(detail_rows, sim_levels)

    print(f"\n{'=' * 70}")
    print("STAGE ANALYSIS (for 60000-sim reasonability)")
    print(f"{'=' * 70}")
    print(
        f"{'Sims':<10} {'AvgTime(s)':<14} {'AvgScore':<14} {'Mean|dScore|':<14} {'Max|dScore|':<14} {'AvgTimeRatio':<14}"
    )
    print("-" * 70)
    for row in summary_rows:
        avg_time = "N/A" if row["avg_elapsed_seconds"] is None else f"{float(row['avg_elapsed_seconds']):.6f}"
        avg_score = "N/A" if row["avg_score"] is None else f"{float(row['avg_score']):.6f}"
        mean_delta = (
            "N/A"
            if row["mean_abs_score_delta_vs_prev"] is None
            else f"{float(row['mean_abs_score_delta_vs_prev']):.6f}"
        )
        max_delta = (
            "N/A"
            if row["max_abs_score_delta_vs_prev"] is None
            else f"{float(row['max_abs_score_delta_vs_prev']):.6f}"
        )
        time_ratio = (
            "N/A"
            if row["avg_time_ratio_vs_prev"] is None
            else f"{float(row['avg_time_ratio_vs_prev']):.6f}"
        )
        print(
            f"{int(row['simulations']):<10} {avg_time:<14} {avg_score:<14} "
            f"{mean_delta:<14} {max_delta:<14} {time_ratio:<14}"
        )

    detail_csv = script_dir / args.detail_csv
    summary_csv = script_dir / args.summary_csv
    write_detail_csv(detail_csv, detail_rows)
    write_stage_summary_csv(summary_csv, summary_rows)

    print(f"\n{'=' * 70}")
    print(f"Detailed CSV saved: {detail_csv.name}")
    print(f"Stage summary CSV saved: {summary_csv.name}")
    print(f"{'=' * 70}")

    valid_rows = [r for r in detail_rows if str(r["status"]) == "OK"]
    all_pass = all(float(r["elapsed_seconds"]) < args.time_threshold for r in valid_rows) and len(valid_rows) > 0
    if all_pass:
        print(f"[PASS] ALL SUCCESSFUL RUNS ARE BELOW {args.time_threshold:.2f}s")
    else:
        print(f"[WARN] SOME RUNS EXCEEDED {args.time_threshold:.2f}s OR FAILED")

    print(f"{'=' * 70}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
