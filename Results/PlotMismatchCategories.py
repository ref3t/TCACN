#!/usr/bin/env python3
"""
Aggregate generic mismatch categories from *_diff_locations columns.

Useful for visualizing the distribution of non-OT predicate mismatches for a
given experiment (e.g., GPT+_TC-CAN_With_File_Different_Session).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DATASET = Path("Data") / "FFFFFFinalResults.xlsx"
DEFAULT_EXPERIMENT = "GPT+_TC-CAN_With_File_Different_Session"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot predicate mismatch counts using *_diff_locations columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Excel workbook with *_diff_locations columns.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="Experiment column prefix (e.g., GPT+_TC-CAN_With_File_Different_Session).",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("Results") / "diff_location_category_counts.xlsx",
        help="Where to store the aggregated mismatch counts.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("Results") / "diff_location_category_counts.png",
        help="Where to save the mismatch category plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the matplotlib window after saving the figure.",
    )
    parser.add_argument(
        "--y-tick-step",
        type=int,
        default=5,
        help="Spacing between Y-axis ticks.",
    )
    parser.add_argument(
        "--y-tick-start",
        type=int,
        default=0,
        help="Starting value for Y-axis ticks.",
    )
    parser.add_argument(
        "--ignore-ot",
        action="store_true",
        help="Skip mismatch entries where either LLM or GT value equals OT.",
    )
    return parser.parse_args()


def tokenize_locations(value: object, ignore_ot: bool) -> List[str]:
    """
    Split a diff_locations cell and return the predicate labels (e.g., BC:RO).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):  # type: ignore[arg-type]
        return []
    labels: List[str] = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        predicate = None
        llm_value = None
        gt_value = None
        colon_parts = token.split(":", 2)
        if len(colon_parts) >= 3:
            predicate = f"{colon_parts[1]}:{colon_parts[2].split()[0]}"
        elif len(colon_parts) >= 2:
            predicate = colon_parts[1].split()[0]
        remainder = token
        if "LLM=" in remainder:
            llm_value = remainder.split("LLM=", 1)[1].split(",", 1)[0].strip().strip(")")
        if "GT=" in remainder:
            gt_value = remainder.split("GT=", 1)[1].split(")", 1)[0].strip()
        if ignore_ot and ("OT" in {llm_value, gt_value}):
            continue
        if predicate:
            labels.append(predicate)
    return labels


def plot_counts(
    categories: List[str],
    counts: List[int],
    output_path: Path,
    show: bool,
    y_tick_step: int,
    y_tick_start: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(categories)), counts, color="#999999", edgecolor="black")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Mismatch count across logs")
    ax.set_title("Predicate mismatch counts")
    for bar, value in zip(bars, counts):
        ax.annotate(
            f"{value}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    if y_tick_step > 0:
        max_count = max(counts) if counts else 0
        upper_bound = ((max_count + y_tick_step - 1) // y_tick_step) * y_tick_step
        start = max(0, y_tick_start)
        ticks = list(range(start, upper_bound + y_tick_step, y_tick_step))
        if start != 0 and 0 not in ticks:
            ticks = [0] + ticks
        ax.set_yticks(ticks)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = pd.read_excel(args.dataset)
    column_name = f"{args.experiment}_diff_locations"
    if column_name not in df.columns:
        raise SystemExit(f"Column missing: {column_name}")

    counter: Counter[str] = Counter()
    for value in df[column_name]:
        counter.update(tokenize_locations(value, args.ignore_ot))
    if not counter:
        raise SystemExit("No mismatch data was found to summarize.")

    records = [
        {"Predicate": predicate, "Count": count}
        for predicate, count in counter.most_common()
    ]
    summary_df = pd.DataFrame(records)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.table_output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False)

    plot_counts(
        summary_df["Predicate"].tolist(),
        summary_df["Count"].tolist(),
        args.figure_output,
        show=args.show,
        y_tick_step=args.y_tick_step,
        y_tick_start=args.y_tick_start,
    )
    print(f"[info] Wrote mismatch counts to {args.table_output}")
    print(f"[info] Saved mismatch plot to {args.figure_output}")


if __name__ == "__main__":
    main()
