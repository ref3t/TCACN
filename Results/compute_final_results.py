#!/usr/bin/env python3
"""
Compute statistical analysis metrics for the FinalResultsRQ2.xlsx workbook.

The script compares each experiment column against the ground truth ACN vector,
derives row-level Hamming similarities, aggregates the totals, and exports the
results to an Excel workbook.
"""

from __future__ import annotations

import argparse
from collections import Counter
from math import asin, sqrt
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

try:  # SciPy renamed binom_test -> binomtest.
    from scipy.stats import binomtest as binom_test
except ImportError:  # pragma: no cover - fallback for older SciPy builds.
    from scipy.stats import binom_test  # type: ignore


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
OT_EQUIVALENT_VALUES = {"OT", "NO", "UN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform the statistical analysis for ACN vectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Data") / "FinalResultsRQ2.xlsx",
        help="Input Excel file that contains the ground truth and GPT+ results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Results") / "FinalResultsRQ2_summary.xlsx",
        help="Path of the workbook where the results will be stored.",
    )
    parser.add_argument(
        "--detailed-output",
        type=Path,
        default=Path("Data") / "FinalResultsRQ2.xlsx",
        help="Excel file under Data/ that stores per-row metrics.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_COLUMNS),
        help="Experiment columns to analyze and compare to the ground truth.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Read the Excel workbook, falling back to a temporary copy or siblings if needed.
    """
    fallback_candidates = [
        dataset_path.with_name(f"{dataset_path.stem} - Copy{dataset_path.suffix}"),
        dataset_path.with_name(f"{dataset_path.stem}_copy{dataset_path.suffix}"),
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

    for fallback in fallback_candidates:
        if fallback.exists():
            print(
                f"[warn] Permission denied for {dataset_path}, attempting to read {fallback} instead."
            )
            return pd.read_excel(fallback)
    raise PermissionError(f"Unable to access {dataset_path} or any fallback copies.")


def normalize_vector(value: object) -> List[str]:
    """
    Split an ACN vector cell into 17 categorical tokens.

    Cells use either ';' or whitespace as delimiters depending on the source.
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


def split_label_value(token: str, fallback_label: str | None = None) -> Tuple[str, str]:
    """
    Split an ACN token into the category label and value (e.g., 'BC:IM', 'NO').
    """
    if token == MISSING_LABEL:
        return (fallback_label or MISSING_LABEL, MISSING_LABEL)
    if "-" in token:
        label, value = token.split("-", 1)
    else:
        label = fallback_label or token
        value = token if fallback_label is None else ""
    return label, value


def is_ot_equivalent(value: str) -> bool:
    """
    Treat OT/NO/UN as equivalent OT-related labels.
    """
    return value.strip().upper() in OT_EQUIVALENT_VALUES if value else False


def count_ot_equivalent_entries(vector: Sequence[str]) -> int:
    """
    Count how many OT-equivalent entries appear within a vector.
    """
    total = 0
    for token in vector:
        _, value = split_label_value(token)
        if is_ot_equivalent(value):
            total += 1
    return total


def ensure_vectors(
    series: Sequence[object], column_name: str | None = None
) -> tuple[list[list[str]], int]:
    """
    Convert an entire pandas Series of ACN vectors into a list of tokens.

    Returns the normalized vectors and the number of rows that were missing a prediction.
    """
    vectors: list[list[str]] = []
    missing = 0
    for row_idx, value in enumerate(series, start=1):
        try:
            tokens = normalize_vector(value)
        except ValueError as exc:
            column_hint = f" for column '{column_name}'" if column_name else ""
            print(f"[warn] {exc} at row {row_idx}{column_hint}; substituting placeholders.")
            tokens = []
        if not tokens:
            tokens = [MISSING_LABEL] * EXPECTED_VECTOR_LENGTH
            missing += 1
        vectors.append(tokens)
    return vectors, missing


def flatten(vectors: Iterable[Sequence[str]]) -> list[str]:
    """Return a flat list of tokens for frequency calculations."""
    flattened: list[str] = []
    for vector in vectors:
        flattened.extend(vector)
    return flattened


def compute_expected_match_probability(
    gt_tokens: Counter[str], pred_tokens: Counter[str], total_positions: int
) -> float:
    """
    Estimate the chance-level match probability p0 given the empirical distributions.
    """
    probability = 0.0
    all_labels = set(gt_tokens) | set(pred_tokens)
    for label in all_labels:
        p_gt = gt_tokens[label] / total_positions
        p_pred = pred_tokens[label] / total_positions
        probability += p_gt * p_pred
    return probability


def interpret_effect_size(h_value: float) -> str:
    magnitude = abs(h_value)
    if magnitude >= 0.80:
        return "large"
    if magnitude >= 0.50:
        return "medium"
    if magnitude >= 0.20:
        return "small"
    return "negligible"


def build_sheet_name(base_name: str, existing: set[str]) -> str:
    """
    Sanitize and deduplicate Excel sheet names (max length 31, no invalid chars).
    """
    invalid_chars = set('[]:*?/\\')
    sanitized = "".join("_" if ch in invalid_chars else ch for ch in base_name).strip()
    if not sanitized:
        sanitized = "Sheet"
    sanitized = sanitized[:31]
    candidate = sanitized
    suffix = 1
    while candidate in existing:
        suffix_str = f"_{suffix}"
        candidate = (sanitized[: 31 - len(suffix_str)] + suffix_str) if len(sanitized) + len(suffix_str) > 31 else sanitized + suffix_str
        suffix += 1
    existing.add(candidate)
    return candidate


def build_row_metrics(
    experiment: str, gt_vectors: Sequence[Sequence[str]], pred_vectors: Sequence[Sequence[str]]
) -> Tuple[Dict[str, List], List[int]]:
    """
    Compute per-row statistics and formatted strings for a given experiment.
    """
    diff_counts: List[int] = []
    diff_locations: List[str] = []
    highlighted_rows: List[str] = []
    ot_mismatch_counts: List[int] = []
    hamming_scores: List[float] = []
    matches_per_row: List[int] = []
    ot_equivalent_counts: List[int] = []
    ot_equivalent_mismatch_counts: List[int] = []
    ot_equivalent_mismatch_details: List[str] = []
    ot_equivalent_mismatch_categories: List[str] = []

    for row_idx, (gt_vec, pred_vec) in enumerate(zip(gt_vectors, pred_vectors), start=1):
        row_diff = 0
        row_locations: List[str] = []
        row_highlights: List[str] = []
        row_ot_mismatches = 0
        row_pred_ot_equivalent = 0
        row_ot_equivalent_mismatches = 0
        row_ot_equivalent_details: List[str] = []
        row_ot_equivalent_categories: List[str] = []

        for pos_idx, (gt_token, pred_token) in enumerate(zip(gt_vec, pred_vec), start=1):
            gt_label, gt_value = split_label_value(gt_token)
            pred_label, pred_value = split_label_value(pred_token, fallback_label=gt_label)
            pred_is_ot_equivalent = is_ot_equivalent(pred_value)
            gt_is_ot_equivalent = is_ot_equivalent(gt_value)
            if pred_is_ot_equivalent:
                row_pred_ot_equivalent += 1
            match = gt_value == pred_value

            if match:
                row_highlights.append(f"{gt_label}-{pred_value}")
            else:
                row_diff += 1
                row_locations.append(
                    f"{pos_idx}:{gt_label} (LLM={pred_value}, GT={gt_value})"
                )
                row_highlights.append(f"{gt_label}:[{pred_value}|GT={gt_value}]")
                if pred_value == "OT":
                    row_ot_mismatches += 1
                if gt_is_ot_equivalent or pred_is_ot_equivalent:
                    row_ot_equivalent_mismatches += 1
                    row_ot_equivalent_details.append(
                        f"{pos_idx}:{gt_label} (LLM={pred_value}, GT={gt_value})"
                    )
                    category_label = gt_label if gt_label != MISSING_LABEL else pred_label
                    row_ot_equivalent_categories.append(category_label)

        diff_counts.append(row_diff)
        diff_locations.append("; ".join(row_locations))
        ot_mismatch_counts.append(row_ot_mismatches)
        ot_equivalent_counts.append(row_pred_ot_equivalent)
        ot_equivalent_mismatch_counts.append(row_ot_equivalent_mismatches)
        ot_equivalent_mismatch_details.append("; ".join(row_ot_equivalent_details))
        ot_equivalent_mismatch_categories.append("; ".join(row_ot_equivalent_categories))
        matches = EXPECTED_VECTOR_LENGTH - row_diff
        matches_per_row.append(matches)
        hamming_scores.append(matches / EXPECTED_VECTOR_LENGTH)
        highlighted_rows.append(" ".join(row_highlights))

    metrics = {
        f"{experiment}_diff_count": diff_counts,
        f"{experiment}_diff_locations": diff_locations,
        f"{experiment}_highlighted": highlighted_rows,
        f"{experiment}_ot_mismatch_count": ot_mismatch_counts,
        f"{experiment}_hamming": hamming_scores,
        f"{experiment}_ot_equivalent_count": ot_equivalent_counts,
        f"{experiment}_ot_equivalent_mismatch_count": ot_equivalent_mismatch_counts,
        f"{experiment}_ot_equivalent_mismatch_details": ot_equivalent_mismatch_details,
        f"{experiment}_ot_equivalent_mismatch_categories": ot_equivalent_mismatch_categories,
    }
    return metrics, matches_per_row


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = load_dataset(args.dataset)
    df = df.dropna(subset=[GROUND_TRUTH_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("No rows with a ground-truth vector were found.")

    gt_vectors, missing_gt = ensure_vectors(df[GROUND_TRUTH_COLUMN], GROUND_TRUTH_COLUMN)
    if missing_gt:
        raise SystemExit("Ground truth column contains missing vectors, cannot proceed.")
    gt_ot_equivalent_counts = [count_ot_equivalent_entries(vector) for vector in gt_vectors]

    total_rows = len(gt_vectors)
    total_positions = total_rows * EXPECTED_VECTOR_LENGTH
    gt_counter = Counter(flatten(gt_vectors))

    summary_records = []
    row_records = []
    detailed_columns: Dict[str, List] = {}
    processed_experiments: List[str] = []
    experiment_metric_columns: Dict[str, List[str]] = {}

    for experiment in args.experiments:
        if experiment not in df.columns:
            print(f"[warn] Column not found, skipping: {experiment}")
            continue
        processed_experiments.append(experiment)
        pred_vectors, missing_rows = ensure_vectors(df[experiment], experiment)
        row_metrics, matches_per_row = build_row_metrics(experiment, gt_vectors, pred_vectors)
        for column_name, values in row_metrics.items():
            detailed_columns[column_name] = values
        experiment_metric_columns[experiment] = list(row_metrics.keys())
        total_matches = sum(matches_per_row)
        pred_counter = Counter(flatten(pred_vectors))
        p0 = compute_expected_match_probability(gt_counter, pred_counter, total_positions)
        p_hat = total_matches / total_positions
        test_result = binom_test(total_matches, total_positions, p0, alternative="two-sided")
        h_value = 2 * (asin(sqrt(p_hat)) - asin(sqrt(p0)))
        effect_size = interpret_effect_size(h_value)

        summary_records.append(
            {
                "Experiment": experiment,
                "K (matches)": total_matches,
                "N (total positions)": total_positions,
                "Hamming accuracy": p_hat,
                "p0 (expected matches)": p0,
                "p-value (binomial test)": test_result.pvalue,
                "Cohen's h": h_value,
                "Effect size": effect_size,
                "Missing vectors": missing_rows,
            }
        )

        for idx, matches in enumerate(matches_per_row, start=1):
            time_value = df.loc[idx - 1, "_time"] if "_time" in df.columns else None
            attack_value = df.loc[idx - 1, "attack"] if "attack" in df.columns else None
            row_records.append(
                {
                    "Experiment": experiment,
                    "Row": idx,
                    "_time": time_value,
                    "attack": attack_value,
                    "matches": matches,
                    "total_positions": EXPECTED_VECTOR_LENGTH,
                    "hamming_similarity": matches / EXPECTED_VECTOR_LENGTH,
                }
            )

    if not summary_records:
        raise SystemExit("No experiment columns were processed successfully.")

    summary_df = pd.DataFrame(summary_records)
    rows_df = pd.DataFrame(row_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="ExperimentSummary", index=False)
        rows_df.to_excel(writer, sheet_name="RowByRow", index=False)

    if processed_experiments:
        base_info_columns = [
            col
            for col in ["_time", "attack", "prompt", GROUND_TRUTH_COLUMN]
            if col in df.columns
        ]
        all_sheet_columns = base_info_columns + [
            col for col in processed_experiments if col in df.columns
        ]
        # Deduplicate while preserving order.
        seen = {}
        for column in all_sheet_columns:
            seen.setdefault(column, None)
        ordered_all_columns = list(seen.keys())

        detailed_df = df[ordered_all_columns].copy()
        gt_ot_column = f"{GROUND_TRUTH_COLUMN}_ot_equivalent_count"
        insert_at = detailed_df.columns.get_loc(GROUND_TRUTH_COLUMN) + 1
        detailed_df.insert(insert_at, gt_ot_column, gt_ot_equivalent_counts)
        for column_name, values in detailed_columns.items():
            detailed_df[column_name] = values
        args.detailed_output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(args.detailed_output, engine="openpyxl") as writer:
            detailed_df.to_excel(writer, sheet_name="AllExperiments", index=False)
            used_sheet_names = {"AllExperiments"}
            for experiment in processed_experiments:
                experiment_cols = [experiment] if experiment in df.columns else []
                metric_cols = [
                    column
                    for column in experiment_metric_columns.get(experiment, [])
                    if column in detailed_df.columns
                ]
                subset_cols = base_info_columns + [gt_ot_column] + experiment_cols + metric_cols
                # Remove duplicates while preserving order for subset.
                ordered_subset = list(dict.fromkeys(subset_cols))
                experiment_df = detailed_df[ordered_subset]
                safe_sheet_name = build_sheet_name(experiment, used_sheet_names)
                experiment_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        print(f"[info] Saved detailed per-row metrics to {args.detailed_output}")

    print(f"[info] Processed {len(summary_records)} experiments over {total_rows} rows.")
    print(f"[info] Saved analysis workbook to {args.output}")


if __name__ == "__main__":
    main()
