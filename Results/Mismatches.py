#!/usr/bin/env python3
"""
Build a boxplot for Ground Truth OT counts, GPT OT counts, and mismatches.

Each series is treated as an experiment and rendered as a separate box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import pandas as pd


# =============================
# Data (hard-coded)
# =============================

GROUND_TRUTH = [
    13,8,13,13,13,13,8,13,8,7,13,13,8,8,13,10,8,10,14,10,10,8,8,9,7,13,13,13,8,8,8,14,14,14,14,14,14,7,7,8
]

GPT_COUNT = [
    15,14,14,14,15,15,15,14,14,15,14,13,14,15,14,17,13,15,15,16,14,13,13,13,13,15,14,13,15,15,16,16,16,16,14,15,17,16,16,15
]

MISMATCH = [
    5,8,5,5,4,4,8,5,8,10,6,6,7,10,6,9,7,8,3,8,8,8,8,6,7,4,5,4,9,9,10,2,2,3,2,4,10,11,10,10
]


SERIES_MAP = {
    "GroundTruth_OT_Count": pd.Series(GROUND_TRUTH),
    "GPT_OT_Count": pd.Series(GPT_COUNT),
    "GroundTruth_GPT_OT_Mismatch_Count": pd.Series(MISMATCH),
}


# Order of appearance in plot
ORDERED_SERIES = [
    "GroundTruth_OT_Count",
    "GPT_OT_Count",
    "GroundTruth_GPT_OT_Mismatch_Count",
]


# =============================
# CLI
# =============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot OT count distributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Results"),
        help="Directory where the plots will be saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive display (useful on servers).",
    )
    return parser.parse_args()


# =============================
# Plot Function
# =============================

def plot_boxplot(args: argparse.Namespace) -> None:

    args.output.mkdir(parents=True, exist_ok=True)

    series_list = [SERIES_MAP[key].dropna() for key in ORDERED_SERIES]
    labels = ORDERED_SERIES

    plt.figure(figsize=(12, 6))

    plt.boxplot(
        series_list,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        meanprops={"color": "#d62728", "linewidth": 2},
        medianprops={"color": "#1f77b4", "linewidth": 2},
    )

    # compute limits
    max_value = max(s.max() for s in series_list)
    min_value = min(s.min() for s in series_list)
    margin = max(0.05 * (max_value - min_value), 0.05)

    ax = plt.gca()
    ax.set_ylim(min_value - margin, max_value + margin)

    # write mean + median labels
    for idx, (key, values) in enumerate(zip(labels, series_list), start=1):
        mean_val = values.mean()
        median_val = values.median()

        ax.text(
            idx + 0.25,
            mean_val,
            f"{mean_val:.2f}",
            ha="left",
            va="center",
            color="#d62728",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            idx - 0.25,
            median_val,
            f"{median_val:.2f}",
            ha="right",
            va="center",
            color="#1f77b4",
            fontsize=9,
            fontweight="bold",
        )

    plt.title("OT Equivalent Counts and Mismatches")
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha="right")

    mean_line = mlines.Line2D([], [], color="#d62728", linewidth=2, label="Mean")
    median_line = mlines.Line2D([], [], color="#1f77b4", linewidth=2, label="Median")
    plt.legend(handles=[mean_line, median_line], loc="best")

    plt.tight_layout()

    out_path = args.output / "boxplot_ot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[info] Saved plot to {out_path}")

    # summary stats
    print("\n[stats]")
    for key, values in zip(labels, series_list):
        print(f" - {key}: mean={values.mean():.3f}, median={values.median():.3f}")

    if not args.no_show:
        plt.show()


# =============================
# Main
# =============================

def main() -> None:
    args = parse_args()
    plot_boxplot(args)


if __name__ == "__main__":
    main()
