#!/usr/bin/env python3
"""
Add OT-equivalent statistics to Data/finalResultsRQ2.xlsx.

The workbook is expected to contain two columns:
    - TC-CAN_GroundTruth
    - GPTTC_ACN

For each row we:
    * Count how many OT-equivalent entries (OT/NO/UN) the ground truth vector has.
    * Count the same for the GPT vector.
    * Record OT-only mismatches (count, detailed description, category list).

All results are written back into the same Excel file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

GROUND_TRUTH_COLUMN = "TC-CAN_GroundTruth"
GPT_COLUMN = "GPTTC_ACN"
EXPECTED_VECTOR_LENGTH = 17
MISSING_LABEL = "MISSING"
OT_EQUIVALENT_VALUES = {"OT", "NO", "UN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute OT-equivalent statistics for FinalResultsRQ2.xlsx.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Data") / "finalResultsRQ2.xlsx",
        help="Excel file that contains ground truth and GPT vectors.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load the workbook, falling back to a temporary copy when OneDrive locks it.
    """
    try:
        return pd.read_excel(dataset_path)
    except PermissionError:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / dataset_path.name
            shutil.copy2(dataset_path, tmp_path)
            return pd.read_excel(tmp_path)


def normalize_vector(value: object) -> List[str]:
    """
    Normalize a cell into a list of tokens, enforcing 17 entries.
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


def ensure_vectors(series: Sequence[object]) -> list[list[str]]:
    """
    Convert the entire column into normalized vectors, filling missing rows.
    """
    vectors: list[list[str]] = []
    for value in series:
        tokens = normalize_vector(value)
        if not tokens:
            tokens = [MISSING_LABEL] * EXPECTED_VECTOR_LENGTH
        vectors.append(tokens)
    return vectors


def split_label_value(token: str, fallback_label: str | None = None) -> Tuple[str, str]:
    """
    Split an ACN token into label and value parts.
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
    Treat OT, NO, and UN as the same OT-equivalent status.
    """
    return value.strip().upper() in OT_EQUIVALENT_VALUES if value else False


def analyze_vectors(
    gt_vectors: Sequence[Sequence[str]], gpt_vectors: Sequence[Sequence[str]]
) -> Tuple[List[int], List[int], List[int], List[str], List[str]]:
    """
    Produce OT counts and mismatch details for aligned vectors.
    """
    gt_counts: List[int] = []
    gpt_counts: List[int] = []
    mismatch_counts: List[int] = []
    mismatch_details: List[str] = []
    mismatch_categories: List[str] = []

    for row_idx, (gt_vec, gpt_vec) in enumerate(zip(gt_vectors, gpt_vectors), start=1):
        gt_total = 0
        gpt_total = 0
        row_mismatch_count = 0
        row_mismatch_details: List[str] = []
        row_mismatch_categories: List[str] = []

        for pos_idx, (gt_token, gpt_token) in enumerate(zip(gt_vec, gpt_vec), start=1):
            gt_label, gt_value = split_label_value(gt_token)
            gpt_label, gpt_value = split_label_value(gpt_token, fallback_label=gt_label)
            gt_is_ot = is_ot_equivalent(gt_value)
            gpt_is_ot = is_ot_equivalent(gpt_value)
            if gt_is_ot:
                gt_total += 1
            if gpt_is_ot:
                gpt_total += 1

            if (gt_value != gpt_value) and (gt_is_ot or gpt_is_ot):
                row_mismatch_count += 1
                row_mismatch_details.append(
                    f"{pos_idx}:{gt_label} (GPT={gpt_value}, GT={gt_value})"
                )
                category_label = gt_label if gt_label != MISSING_LABEL else gpt_label
                row_mismatch_categories.append(category_label)

        gt_counts.append(gt_total)
        gpt_counts.append(gpt_total)
        mismatch_counts.append(row_mismatch_count)
        mismatch_details.append("; ".join(row_mismatch_details))
        mismatch_categories.append("; ".join(row_mismatch_categories))

    return gt_counts, gpt_counts, mismatch_counts, mismatch_details, mismatch_categories


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = load_dataset(args.dataset)
    missing_columns = [col for col in (GROUND_TRUTH_COLUMN, GPT_COLUMN) if col not in df.columns]
    if missing_columns:
        raise SystemExit(f"Missing required column(s): {', '.join(missing_columns)}")

    gt_vectors = ensure_vectors(df[GROUND_TRUTH_COLUMN])
    gpt_vectors = ensure_vectors(df[GPT_COLUMN])

    (
        gt_counts,
        gpt_counts,
        mismatch_counts,
        mismatch_details,
        mismatch_categories,
    ) = analyze_vectors(gt_vectors, gpt_vectors)

    df[f"{GROUND_TRUTH_COLUMN}_ot_equivalent_count"] = gt_counts
    df[f"{GPT_COLUMN}_ot_equivalent_count"] = gpt_counts
    df[f"{GPT_COLUMN}_ot_equivalent_mismatch_count"] = mismatch_counts
    df[f"{GPT_COLUMN}_ot_equivalent_mismatch_details"] = mismatch_details
    df[f"{GPT_COLUMN}_ot_equivalent_mismatch_categories"] = mismatch_categories

    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.dataset, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    print(f"[info] Saved OT analysis to {args.dataset}")


if __name__ == "__main__":
    main()
