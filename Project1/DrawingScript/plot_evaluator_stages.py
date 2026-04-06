#!/usr/bin/env python3
"""
Plot evaluator multi-stage experiment figures.

Input CSV files (default):
- ../Project1/Project1/Testcase/Evaluator/timing_score_stages.csv
- ../Project1/Project1/Testcase/Evaluator/timing_score_stage_summary.csv

Output directory (default):
- ./figures/evaluator_multistage
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _to_int(value: str) -> int:
    return int(value.strip())


def _to_float(value: str) -> float:
    return float(value.strip())


def read_stage_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "case": r["case"],
                    "simulations": _to_int(r["simulations"]),
                    "elapsed_seconds": _to_float(r["elapsed_seconds"]),
                    "score": _to_float(r["score"]),
                    "status": r.get("status", ""),
                }
            )
    return rows


def read_summary_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "simulations": _to_int(r["simulations"]),
                    "avg_elapsed_seconds": _to_float(r["avg_elapsed_seconds"]),
                    "avg_score": _to_float(r["avg_score"]),
                    "mean_abs_score_delta_vs_prev": _to_float(r["mean_abs_score_delta_vs_prev"]) if r["mean_abs_score_delta_vs_prev"].strip() else None,
                    "max_abs_score_delta_vs_prev": _to_float(r["max_abs_score_delta_vs_prev"]) if r["max_abs_score_delta_vs_prev"].strip() else None,
                    "avg_time_ratio_vs_prev": _to_float(r["avg_time_ratio_vs_prev"]) if r["avg_time_ratio_vs_prev"].strip() else None,
                }
            )
    return rows


def group_by_case(stage_rows: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for r in stage_rows:
        grouped.setdefault(r["case"], []).append(r)
    for case_name in grouped:
        grouped[case_name].sort(key=lambda x: x["simulations"])
    return grouped


def make_case_runtime_plot(grouped: Dict[str, List[dict]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5), dpi=140)
    for case_name, rows in sorted(grouped.items()):
        x = [r["simulations"] for r in rows]
        y = [r["elapsed_seconds"] for r in rows]
        plt.plot(x, y, marker="o", linewidth=2, label=case_name)

    plt.title("Evaluator Runtime by Simulation Stage")
    plt.xlabel("Simulations")
    plt.ylabel("Elapsed Time (s)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_case_score_plot(grouped: Dict[str, List[dict]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5), dpi=140)
    for case_name, rows in sorted(grouped.items()):
        x = [r["simulations"] for r in rows]
        y = [r["score"] for r in rows]
        plt.plot(x, y, marker="o", linewidth=2, label=case_name)

    plt.title("Evaluator Score by Simulation Stage")
    plt.xlabel("Simulations")
    plt.ylabel("Score")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_summary_dual_axis_plot(summary_rows: List[dict], out_path: Path) -> None:
    x = [r["simulations"] for r in summary_rows]
    avg_time = [r["avg_elapsed_seconds"] for r in summary_rows]
    avg_score = [r["avg_score"] for r in summary_rows]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=140)
    line1 = ax1.plot(x, avg_time, marker="o", linewidth=2, color="#1f77b4", label="Avg Time (s)")
    ax1.set_xlabel("Simulations")
    ax1.set_ylabel("Avg Elapsed Time (s)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    line2 = ax2.plot(x, avg_score, marker="s", linewidth=2, color="#d62728", label="Avg Score")
    ax2.set_ylabel("Avg Score", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Stage Summary: Time vs Score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _extract_non_null(summary_rows: List[dict], key: str) -> Tuple[List[int], List[float]]:
    x: List[int] = []
    y: List[float] = []
    for r in summary_rows:
        if r[key] is not None:
            x.append(r["simulations"])
            y.append(r[key])
    return x, y


def make_stability_plot(summary_rows: List[dict], out_path: Path) -> None:
    x_mean, mean_delta = _extract_non_null(summary_rows, "mean_abs_score_delta_vs_prev")
    x_max, max_delta = _extract_non_null(summary_rows, "max_abs_score_delta_vs_prev")

    plt.figure(figsize=(8, 5), dpi=140)
    if x_mean:
        plt.plot(x_mean, mean_delta, marker="o", linewidth=2, label="Mean |Delta Score|")
    if x_max:
        plt.plot(x_max, max_delta, marker="s", linewidth=2, label="Max |Delta Score|")

    plt.title("Score Stability Across Stages")
    plt.xlabel("Simulations")
    plt.ylabel("Absolute Score Change vs Previous Stage")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_time_ratio_plot(summary_rows: List[dict], out_path: Path) -> None:
    x, ratio = _extract_non_null(summary_rows, "avg_time_ratio_vs_prev")

    plt.figure(figsize=(8, 5), dpi=140)
    plt.bar([str(v) for v in x], ratio, color="#2ca02c", alpha=0.85)
    plt.title("Average Time Growth Ratio vs Previous Stage")
    plt.xlabel("Simulations")
    plt.ylabel("Avg Time Ratio")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_stages = script_dir.parent / "Project1" / "Project1" / "Testcase" / "Evaluator" / "timing_score_stages.csv"
    default_summary = script_dir.parent / "Project1" / "Project1" / "Testcase" / "Evaluator" / "timing_score_stage_summary.csv"
    default_output = script_dir / "figures" / "evaluator_multistage"

    parser = argparse.ArgumentParser(description="Plot evaluator multi-stage experiment figures")
    parser.add_argument("--stages-csv", type=Path, default=default_stages, help="Path to timing_score_stages.csv")
    parser.add_argument("--summary-csv", type=Path, default=default_summary, help="Path to timing_score_stage_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=default_output, help="Directory for saved plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.stages_csv.exists():
        raise FileNotFoundError(f"Stages CSV not found: {args.stages_csv}")
    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stage_rows = [r for r in read_stage_rows(args.stages_csv) if r["status"] == "OK"]
    summary_rows = read_summary_rows(args.summary_csv)
    grouped = group_by_case(stage_rows)

    make_case_runtime_plot(grouped, args.out_dir / "evaluator_case_runtime.png")
    make_case_score_plot(grouped, args.out_dir / "evaluator_case_score.png")
    make_summary_dual_axis_plot(summary_rows, args.out_dir / "evaluator_summary_time_score.png")
    make_stability_plot(summary_rows, args.out_dir / "evaluator_summary_score_stability.png")
    make_time_ratio_plot(summary_rows, args.out_dir / "evaluator_summary_time_ratio.png")

    print("Saved figures:")
    print(f"- {args.out_dir / 'evaluator_case_runtime.png'}")
    print(f"- {args.out_dir / 'evaluator_case_score.png'}")
    print(f"- {args.out_dir / 'evaluator_summary_time_score.png'}")
    print(f"- {args.out_dir / 'evaluator_summary_score_stability.png'}")
    print(f"- {args.out_dir / 'evaluator_summary_time_ratio.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
