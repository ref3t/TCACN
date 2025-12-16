#!/usr/bin/env python3
"""
Analyze OT counts for TC-ACN vectors across ground truth and LLM configurations.

Outputs per-row OT counts, summary statistics, ΔOT comparisons, and predicate-level
breakdowns to an Excel workbook (RQ2Results.xlx).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Sequence, Tuple

import pandas as pd


GROUND_TRUTH_COLUMN = "TC-CAN_GroundTruth"
EXPERIMENT_COLUMNS = [
    "GPT+_TC-CAN_With_File_Same_Session",
    "GPT+_TC-CAN_Without_File_Same_Session",
    "GPT+_TC-CAN_Without_File_Different_Session",
    "GPT+_TC-CAN_With_File_Different_Session",
    "LLM Free ( Same session)",
    "LLM Free ( Different session)",
]
EXPECTED_VECTOR_LENGTH = 17
MISSING_LABEL = "MISSING"
OUTPUT_PATH = Path("Data") / "RQ2Results.xlx"
BASE_PREDICATE_FAMILIES = ["BC", "TT", "TA", "AC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute OT statistics for GT and LLM TC-ACN vectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Data") / "GTAttacksLogsFinal.xlsx",
        help="Excel file with ground truth and LLM vectors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Excel workbook where the RQ2 results will be saved.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load the Excel workbook, copying to a temporary file if OneDrive locks it.
    """
    fallback_candidates = [
        dataset_path.with_name(f"{dataset_path.stem}_copy{dataset_path.suffix}"),
        dataset_path.with_name(f"{dataset_path.stem} - Copy{dataset_path.suffix}"),
        dataset_path.with_name(f"{dataset_path.stem}_temp{dataset_path.suffix}"),
    ]
    try:
        return pd.read_excel(dataset_path)
    except PermissionError:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / dataset_path.name
            try:
                shutil.copy2(dataset_path, tmp_path)
                return pd.read_excel(tmp_path)
            except PermissionError:
                pass
    for candidate in fallback_candidates:
        if candidate.exists():
            print(
                f"[warn] Unable to access {dataset_path}, using fallback {candidate}."
            )
            return pd.read_excel(candidate)
    raise PermissionError(f"Unable to open {dataset_path} or any fallback copy.")


def normalize_vector(value: object) -> List[str]:
    """
    Normalize a TC-ACN vector cell to a list of tokens.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):  # type: ignore[arg-type]
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.replace("\n", " ").replace("\r", " ")
    if ";" in text:
        parts = [chunk.strip() for chunk in text.split(";")]
    else:
        parts = [chunk.strip() for chunk in text.split()]
    parts = [chunk for chunk in parts if chunk]
    if parts and len(parts) != EXPECTED_VECTOR_LENGTH:
        raise ValueError(
            f"Expected {EXPECTED_VECTOR_LENGTH} entries, found {len(parts)}: {text}"
        )
    return parts


def ensure_vectors(series: Sequence[object]) -> tuple[list[list[str]], int]:
    vectors: list[list[str]] = []
    missing = 0
    for value in series:
        tokens = normalize_vector(value)
        if not tokens:
            tokens = [MISSING_LABEL] * EXPECTED_VECTOR_LENGTH
            missing += 1
        vectors.append(tokens)
    return vectors, missing


def count_ot_entries(vector: Sequence[str]) -> tuple[int, Counter[str]]:
    """
    Count OT tokens in a vector and how they distribute across predicate families.
    """
    total = 0
    per_prefix: Counter[str] = Counter()
    for token in vector:
        if token.endswith("-OT"):
            total += 1
            prefix = token.split(":", 1)[0] if ":" in token else "Unknown"
            per_prefix[prefix] += 1
    return total, per_prefix


def summarize_counts(counts: List[int]) -> Dict[str, float]:
    series = pd.Series(counts)
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
        "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
    }


def interpret_delta(delta: float) -> str:
    if abs(delta) < 0.25:
        return "Matches GT (semantically aligned)"
    if delta > 0:
        return "More OT than GT -> less complete"
    return "Fewer OT than GT -> more specific"


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = load_dataset(args.dataset)
    df = df.dropna(subset=[GROUND_TRUTH_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("No rows with ground truth vectors were found.")

    gt_vectors, missing_gt = ensure_vectors(df[GROUND_TRUTH_COLUMN])
    if missing_gt:
        raise SystemExit("Ground truth column contains missing vectors.")

    per_row_counts: Dict[str, List[int]] = {}
    per_prefix_counts: Dict[str, Counter[str]] = {}
    summary_stats: Dict[str, Dict[str, float]] = {}

    def process_column(label: str, vectors: Sequence[Sequence[str]]) -> None:
        counts: List[int] = []
        prefix_counter: Counter[str] = Counter()
        for vector in vectors:
            total, per_prefix = count_ot_entries(vector)
            counts.append(total)
            prefix_counter.update(per_prefix)
        per_row_counts[label] = counts
        per_prefix_counts[label] = prefix_counter
        summary_stats[label] = summarize_counts(counts)

    process_column("Ground Truth", gt_vectors)

    detailed_columns: Dict[str, List[str]] = {}
    missing_summary = []

    for experiment in EXPERIMENT_COLUMNS:
        if experiment not in df.columns:
            print(f"[warn] Column missing, skipping: {experiment}")
            continue
        vectors, missing = ensure_vectors(df[experiment])
        if missing:
            missing_summary.append(f"{experiment}: {missing} missing vectors replaced with placeholders.")
        process_column(experiment, vectors)

    gt_mean = summary_stats["Ground Truth"]["mean"]

    summary_records = []
    for label, stats in summary_stats.items():
        delta = stats["mean"] - gt_mean if label != "Ground Truth" else 0.0
        summary_records.append(
            {
                "Experiment": label,
                "Mean_OT": stats["mean"],
                "Median_OT": stats["median"],
                "Q1": stats["q1"],
                "Q3": stats["q3"],
                "IQR": stats["iqr"],
                "Min": stats["min"],
                "Max": stats["max"],
                "Delta_OT_vs_GT": delta,
                "Interpretation": interpret_delta(delta) if label != "Ground Truth" else "Ground truth baseline",
            }
        )

    base_columns = []
    for col in ["_time", "attack", "prompt"]:
        if col in df.columns:
            base_columns.append(col)
    per_row_df = pd.DataFrame({col: df[col] for col in base_columns})
    per_row_df["Ground Truth OT"] = per_row_counts["Ground Truth"]
    for experiment, counts in per_row_counts.items():
        if experiment == "Ground Truth":
            continue
        per_row_df[f"{experiment} OT"] = counts
        per_row_df[f"{experiment} ΔOT"] = [
            exp_count - gt_count for exp_count, gt_count in zip(counts, per_row_counts["Ground Truth"])
        ]

    predicate_records = []
    for experiment, counter in per_prefix_counts.items():
        total_ot = sum(counter.values())
        families = sorted(set(counter) | set(BASE_PREDICATE_FAMILIES))
        for prefix in families:
            count = counter.get(prefix, 0)
            predicate_records.append(
                {
                    "Experiment": experiment,
                    "PredicateFamily": prefix,
                    "OT_Count": count,
                    "Share_of_Experiment_OT": count / total_ot if total_ot else 0.0,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        per_row_df.to_excel(writer, sheet_name="PerRow_OT_Counts", index=False)
        pd.DataFrame(summary_records).to_excel(writer, sheet_name="Summary_Stats", index=False)
        pd.DataFrame(predicate_records).to_excel(writer, sheet_name="Predicate_OT", index=False)
        if missing_summary:
            pd.DataFrame({"Notes": missing_summary}).to_excel(writer, sheet_name="Notes", index=False)

    print(f"[info] Saved RQ2 OT analysis to {args.output}")


if __name__ == "__main__":
    main()
