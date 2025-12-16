#!/usr/bin/env python3
"""
Build a boxplot for Ground Truth OT counts, GPT OT counts, and mismatches.

Each series is treated as an experiment and rendered as a separate box.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import pandas as pd


# =============================
# Configuration
# =============================

DEFAULT_DATASET = Path("Data") / "FFFFFFinalResults.xlsx"
DEFAULT_EXPERIMENT = "GPT+_TC-CAN_With_File_Different_Session"
GROUND_TRUTH_COLUMN = "TC-CAN_GroundTruth_ot_equivalent_count"

FRIENDLY_EXPERIMENT_NAMES = {
    "GPT+_TC-CAN_With_File_Same_Session": "ProChatGPTWithTaxonomyFileWithSameSession",
    "GPT+_TC-CAN_Without_File_Same_Session": "ProChatGPTWithoutTaxonomyFileWithSameSession",
    "GPT+_TC-CAN_Without_File_Different_Session": "ProChatGPTWithoutTaxonomyFileWithDifferentSession",
    "GPT+_TC-CAN_With_File_Different_Session": "ProChatGPTWithTaxonomyFileWithDifferentSession",
    "LLM Free ( Same session)": "FreeChatGPTWithSameSession",
    "LLM Free ( Different session)": "FreeChatGPTWithDifferentSession",
}


# =============================
# CLI
# =============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot OT count distributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Excel file that contains the ground truth and experiment OT statistics.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="Experiment column to visualize (matches the TC-ACN prediction column name).",
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

def extract_series(dataset: Path, experiment: str) -> "OrderedDict[str, pd.Series]":
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    df = pd.read_excel(dataset)

    required_columns = [
        GROUND_TRUTH_COLUMN,
        f"{experiment}_ot_equivalent_count",
        f"{experiment}_ot_equivalent_mismatch_count",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {dataset}: {', '.join(missing)}")

    friendly_name = FRIENDLY_EXPERIMENT_NAMES.get(experiment, experiment)

    return OrderedDict(
        [
            ("Manual_OT_Count", df[GROUND_TRUTH_COLUMN].dropna()),
            (f"LLM_OT_Count", df[f"{experiment}_ot_equivalent_count"].dropna()),
            (
                f"Manual_LLM_OT_Mismatch_Count",
                df[f"{experiment}_ot_equivalent_mismatch_count"].dropna(),
            ),
        ]
    )


def plot_boxplot(args: argparse.Namespace, series_map: "OrderedDict[str, pd.Series]") -> None:

    args.output.mkdir(parents=True, exist_ok=True)

    labels = list(series_map.keys())
    series_list = [series_map[label] for label in labels]

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
    series_map = extract_series(args.dataset, args.experiment)
    plot_boxplot(args, series_map)


if __name__ == "__main__":
    main()
