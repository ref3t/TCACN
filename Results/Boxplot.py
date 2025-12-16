#!/usr/bin/env python3
"""
Build a boxplot from the FinalBoxplot.xlsx workbook.

Each numeric column is treated as an experiment and rendered as a separate box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import pandas as pd


FRIENDLY_NAMES = {
    "LLM Free ( Same session)": "FreeChatGPTWithSameSession",
    "LLM Free ( Different session)": "FreeChatGPTWithDifferentSession",
    "GPT+_TC-CAN_With_File_Same_Session": "ChatGPTPlusWithTaxonomyFileWithSameSession",
    "GPT+_TC-CAN_With_File_Different_Session": "ChatGPTPlusWithTaxonomyFileWithDifferentSession",
    "GPT+_TC-CAN_Without_File_Same_Session": "ChatGPTPlusWithoutTaxonomyFileWithSameSession",
    "GPT+_TC-CAN_Without_File_Different_Session": "ChatGPTPlusWithoutTaxonomyFileWithDifferentSession",
}

SHORT_NAMES = {
    "LLM Free ( Same session)": "LLM Free Same",
    "LLM Free ( Different session)": "LLM Free Diff",
    "GPT+_TC-CAN_With_File_Same_Session": "File+Same",
    "GPT+_TC-CAN_With_File_Different_Session": "File+Diff",
    "GPT+_TC-CAN_Without_File_Same_Session": "NoFile+Same",
    "GPT+_TC-CAN_Without_File_Different_Session": "NoFile+Diff",
}

ORDERED_EXPERIMENTS = [
    "LLM Free ( Same session)",
    "LLM Free ( Different session)",
    "GPT+_TC-CAN_With_File_Same_Session",
    "GPT+_TC-CAN_With_File_Different_Session",
    "GPT+_TC-CAN_Without_File_Same_Session",
    "GPT+_TC-CAN_Without_File_Different_Session",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot experiment distributions stored in the final results workbook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Data") / "FFFFFFinalResults.xlsx",
        help="Excel file that contains one column per experiment.",
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
        help="Skip interactive display (useful on headless servers).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = pd.read_excel(args.dataset)

    diff_series = {
        col.removesuffix("_diff_count"): df[col].dropna()
        for col in df.columns
        if col.endswith("_diff_count")
    }
    hamming_series = {
        col.removesuffix("_hamming"): df[col].dropna()
        for col in df.columns
        if col.endswith("_hamming")
    }

    if not diff_series and not hamming_series:
        raise SystemExit("No *_diff_count or *_hamming columns found to plot.")

    args.output.mkdir(parents=True, exist_ok=True)

    def plot_metric(series_map, title, ylabel, filename):
        if not series_map:
            return
        ordered_keys = [key for key in ORDERED_EXPERIMENTS if key in series_map] or list(series_map.keys())
        plt.figure(figsize=(12, 6))
        data_series = [series_map[key] for key in ordered_keys]
        plt.boxplot(
            data_series,
            tick_labels=[FRIENDLY_NAMES.get(key, key) for key in ordered_keys],
            showmeans=True,
            meanline=True,
            meanprops={"color": "#d62728", "linewidth": 2},
            medianprops={"color": "#1f77b4", "linewidth": 2},
        )
        max_value = max(values.max() for values in data_series)
        min_value = min(values.min() for values in data_series)
        margin = max(0.05 * (max_value - min_value), 0.05)
        ax = plt.gca()
        ax.set_ylim(min_value - margin, max_value + margin)
        stats = []
        for idx, (key, values) in enumerate(zip(ordered_keys, data_series), start=1):
            mean_val = values.mean()
            median_val = values.median()
            stats.append((key, mean_val, median_val))
            ax.text(
                idx + 0.25,
                mean_val,
                f"{mean_val:.3f}",
                ha="left",
                va="center",
                color="#d62728",
                fontsize=9,
                fontweight="bold",
            )
            ax.text(
                idx - 0.25,
                median_val,
                f"{median_val:.3f}",
                ha="right",
                va="center",
                color="#1f77b4",
                fontsize=9,
                fontweight="bold",
            )
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        mean_line = mlines.Line2D([], [], color="#d62728", linewidth=2, label="Mean")
        median_line = mlines.Line2D([], [], color="#1f77b4", linewidth=2, label="Median")
        plt.legend(handles=[mean_line, median_line], loc="best")
        plt.tight_layout()
        out_path = args.output / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved {title} to {out_path}")
        print(f"[stats] {title}")
        for key, mean_val, median_val in stats:
            friendly = FRIENDLY_NAMES.get(key, key)
            short = SHORT_NAMES.get(key, "")
            label = f"{friendly} ({short})" if short else friendly
            print(f" - {label}: mean={mean_val:.4f}, median={median_val:.4f}")
        if not args.no_show:
            plt.show()

    plot_metric(diff_series, "Differences per Experiment", "Mismatched Predicates", "boxplot_diff.png")
    plot_metric(
        hamming_series,
        "Hamming Accuracy per Experiment",
        "Hamming Accuracy",
        "boxplot_hamming.png",
    )

    print("\n[legend] Short experiment names:")
    for column, short in SHORT_NAMES.items():
        friendly = FRIENDLY_NAMES.get(column, column)
        print(f" - {short}: {friendly}")


if __name__ == "__main__":
    main()
