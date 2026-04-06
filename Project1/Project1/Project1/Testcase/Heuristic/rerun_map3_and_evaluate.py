#!/usr/bin/env python3
"""
Rerun heuristic solver for map3 and re-evaluate it, then append one-row CSV.

Usage:
    python rerun_map3_and_evaluate.py

The script runs two steps in order:
1) Heuristic solve for map3
2) Evaluator scoring for map3 result

It prints timing/score for manual report filling and writes one row to CSV.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun map3 and evaluate score")

    # Heuristic run config
    parser.add_argument("--run-network", default="map3/dataset2", help="Heuristic network path")
    parser.add_argument("--run-initial", default="map3/seed2", help="Heuristic initial seed path")
    parser.add_argument("--balanced", default="map3/seed_balanced", help="Heuristic output seed path")
    parser.add_argument("--budget", type=int, default=15, help="Budget k")
    parser.add_argument("--mc-sim", type=int, default=200, help="Heuristic MC scenarios per step")
    parser.add_argument("--max-iter", type=int, default=20, help="Heuristic IMRank max iterations")
    parser.add_argument("--candidate-size", type=int, default=3000, help="Heuristic candidate pool size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for heuristic")

    # Evaluation config
    parser.add_argument(
        "--eval-network",
        default=None,
        help="Evaluator network path (default: same as --run-network)",
    )
    parser.add_argument(
        "--eval-initial",
        default=None,
        help="Evaluator initial seed path (default: same as --run-initial, i.e. map3/seed2)",
    )
    parser.add_argument("--eval-output", default="map3/result.txt", help="Evaluator score output path")
    parser.add_argument("--simulations", type=int, default=60000, help="Evaluator MC simulations")

    # Logging
    parser.add_argument(
        "--summary-csv",
        default="map3/rerun_map3_summary.csv",
        help="CSV file to append one run summary row",
    )
    parser.add_argument(
        "--print-stdout",
        action="store_true",
        help="Print full stdout/stderr of subprocesses",
    )

    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path, print_stdout: bool) -> tuple[int, float, str, str]:
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if print_stdout:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

    return result.returncode, elapsed, result.stdout, result.stderr


def read_score(score_path: Path) -> float:
    text = score_path.read_text(encoding="utf-8").strip()
    return float(text)


def append_csv(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()

    fieldnames = [
        "timestamp",
        "map",
        "budget",
        "simulations",
        "heuristic_elapsed_seconds",
        "evaluation_elapsed_seconds",
        "score",
        "balanced_file",
        "result_file",
        "status",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    eval_network = args.eval_network if args.eval_network is not None else args.run_network
    eval_initial = args.eval_initial if args.eval_initial is not None else args.run_initial

    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.mc_sim <= 0 or args.simulations <= 0 or args.max_iter <= 0:
        raise ValueError("--mc-sim, --simulations, --max-iter must be positive")

    print("=" * 72)
    print("Rerun map3 and Evaluate")
    print("=" * 72)
    print(f"Python: {sys.executable}")
    print(f"Working dir: {script_dir}")
    print(f"Run initial seed: {args.run_initial}")
    print(f"Eval initial seed: {eval_initial}")
    if eval_initial != args.run_initial:
        print("[WARN] Eval initial seed differs from run initial seed.")

    heur_cmd = [
        sys.executable,
        "IEMP_Heur.py",
        "-n",
        args.run_network,
        "-i",
        args.run_initial,
        "-b",
        args.balanced,
        "-k",
        str(args.budget),
        "--mc-sim",
        str(args.mc_sim),
        "--max-iter",
        str(args.max_iter),
        "--candidate-size",
        str(args.candidate_size),
    ]
    if args.seed is not None:
        heur_cmd.extend(["--seed", str(args.seed)])

    print("\n[1/2] Running heuristic (map3)...")
    rc_heur, t_heur, out_heur, err_heur = run_cmd(heur_cmd, script_dir, args.print_stdout)
    if rc_heur != 0:
        print("[ERROR] Heuristic run failed.")
        print(out_heur)
        print(err_heur)
        return 1

    eval_cmd = [
        sys.executable,
        "../Evaluator/Evaluator.py",
        "-n",
        eval_network,
        "-i",
        eval_initial,
        "-b",
        args.balanced,
        "-k",
        str(args.budget),
        "-o",
        args.eval_output,
        "--simulations",
        str(args.simulations),
    ]

    print("\n[2/2] Evaluating map3 result...")
    rc_eval, t_eval, out_eval, err_eval = run_cmd(eval_cmd, script_dir, args.print_stdout)
    if rc_eval != 0:
        print("[ERROR] Evaluation failed.")
        print(out_eval)
        print(err_eval)
        return 1

    score_path = script_dir / args.eval_output
    if not score_path.exists():
        print(f"[ERROR] Score file not found: {score_path}")
        return 1

    score = read_score(score_path)

    print("\n" + "=" * 72)
    print("MAP3 RESULT")
    print("=" * 72)
    print(f"Heuristic runtime (s): {t_heur:.4f}")
    print(f"Evaluation runtime (s): {t_eval:.4f}")
    print(f"Final score: {score:.6f}")
    print("-" * 72)
    print("Report fill hint:")
    print(f"Table A map3 -> Runtime={t_heur:.4f}, Score={score:.6f}")
    print(f"Table C map3 (Heur cols) -> Runtime={t_heur:.4f}, Score={score:.6f}")

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "map": "map3",
        "budget": str(args.budget),
        "simulations": str(args.simulations),
        "heuristic_elapsed_seconds": f"{t_heur:.6f}",
        "evaluation_elapsed_seconds": f"{t_eval:.6f}",
        "score": f"{score:.6f}",
        "balanced_file": args.balanced,
        "result_file": args.eval_output,
        "status": "OK",
    }

    summary_csv_path = script_dir / args.summary_csv
    append_csv(summary_csv_path, row)
    print(f"CSV appended: {summary_csv_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
