#!/usr/bin/env python3
"""
Aggregate OT mismatch categories and plot their frequencies.

Reads Data/finalResultsRQ2.xlsx, counts how often each category appears in
GPTTC_ACN_ot_equivalent_mismatch_categories, saves a summary table, and
generates a color-coded figure.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

CATEGORY_COLUMN = "GPTTC_ACN_ot_equivalent_mismatch_categories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize OT mismatch categories for FinalResultsRQ2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Data") / "finalResultsRQ2.xlsx",
        help="Excel file containing GPTTC_ACN_ot_equivalent_mismatch_categories.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("Results") / "finalResultsRQ2_category_counts.xlsx",
        help="Where to store the aggregated category counts.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("Results") / "finalResultsRQ2_category_counts.png",
        help="Where to save the category frequency plot.",
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
        help="Spacing between Y-axis ticks (use 1 for every unit).",
    )
    parser.add_argument(
        "--y-tick-start",
        type=int,
        default=0,
        help="Starting value for Y-axis ticks (e.g., 2 to show 2,4,6,...).",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Read the Excel workbook, falling back to a temporary copy if locked.
    """
    try:
        return pd.read_excel(dataset_path)
    except PermissionError:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / dataset_path.name
            shutil.copy2(dataset_path, tmp_path)
            return pd.read_excel(tmp_path)


def tokenize_categories(value: object) -> List[str]:
    """
    Split a categories cell into individual tokens.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):  # type: ignore[arg-type]
        return []
    tokens = [token.strip() for token in str(value).split(";")]
    return [token for token in tokens if token]


def plot_counts(
    categories: List[str],
    counts: List[int],
    output_path: Path,
    show: bool = False,
    y_tick_step: int = 5,
    y_tick_start: int = 0,
) -> None:
    """
    Draw a bar chart with one color per category.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    color_map = plt.get_cmap("Greys")
    colors = [color_map(0.4 + 0.5 * (i / max(1, len(categories) - 1))) for i in range(len(categories))]
    positions = range(len(categories))
    bars = ax.bar(positions, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Mismatch count across logs")
    ax.set_title("OT-Equivalent Mismatch Categories")
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
        tick_values = list(range(start, upper_bound + y_tick_step, y_tick_step))
        if start != 0 and 0 not in tick_values:
            tick_values = [0] + tick_values
        ax.set_yticks(tick_values)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = load_dataset(args.dataset)
    if CATEGORY_COLUMN not in df.columns:
        raise SystemExit(f"Column missing: {CATEGORY_COLUMN}")

    counter: Counter[str] = Counter()
    for value in df[CATEGORY_COLUMN]:
        counter.update(tokenize_categories(value))

    if not counter:
        raise SystemExit("No category data was found to summarize.")

    records = [
        {"Category": category, "Count": count}
        for category, count in counter.most_common()
    ]
    summary_df = pd.DataFrame(records)

    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.table_output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False)

    plot_counts(
        summary_df["Category"].tolist(),
        summary_df["Count"].tolist(),
        args.figure_output,
        show=args.show,
        y_tick_step=args.y_tick_step,
        y_tick_start=args.y_tick_start,
    )
    print(f"[info] Wrote category counts to {args.table_output}")
    print(f"[info] Saved category plot to {args.figure_output}")


if __name__ == "__main__":
    main()
