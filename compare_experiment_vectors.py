#!/usr/bin/env python3
"""
Compare TC-CAN experiment vectors against the labelled ground truth.

The script augments the Excel dataset with three columns per experiment:
- <experiment>_diff_count ........ number of predicates that differ from GT
- <experiment>_hamming ............ match ratio (0.0-1.0)
- <experiment>_highlighted ........ experiment vector with mismatches annotated
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


ACN_PREDICATES: Sequence[str] = [
    "BC:IM",
    "BC:RO",
    "BC:SE",
    "TT:AC",
    "TT:AV",
    "TT:FR",
    "TT:BA",
    "TT:DE",
    "TT:IG",
    "TT:MA",
    "TT:SO",
    "TT:VU",
    "TA:AM",
    "AC:IN",
    "AC:OS",
    "AC:PH",
    "AC:VE",
]

DEFAULT_EXPERIMENT_COLUMNS: Sequence[str] = (
    "GPT+_TC-CAN_With_File_Same_Session",
    "GPT+_TC-CAN_Without_File_Same_Session",
    "GPT+_TC-CAN_Without_File_Different_Session",
    "GPT+_TC-CAN_With_File_Different_Session",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare experiment TC-CAN vectors with the ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Data") / "GTAttacksLogsFinal.xlsx",
        help="Excel workbook that contains the ground truth and experiment vectors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to store the augmented workbook. Defaults to --dataset "
            "and overwrites the file when omitted."
        ),
    )
    parser.add_argument(
        "--ground-truth-column",
        default="TC-CAN_GroundTruth",
        help="Column that stores the reference TC-CAN vector.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENT_COLUMNS),
        help="Experiment columns to compare against the ground truth.",
    )
    return parser.parse_args()


def normalize_vector(vector_text: object) -> List[str]:
    """Return the canonical list of predicate tokens for easier comparison."""
    if pd.isna(vector_text):
        vector_text = ""
    text = str(vector_text)
    tokens = [tok.strip() for tok in text.split() if tok.strip()]
    token_map = {}
    for token in tokens:
        if "-" not in token:
            continue
        predicate, value = token.split("-", 1)
        predicate = predicate.strip()
        token_map[predicate] = f"{predicate}-{value.strip()}"
    canonical = [
        token_map.get(predicate, f"{predicate}-MISSING") for predicate in ACN_PREDICATES
    ]
    return canonical


def _value_part(token: str) -> str:
    if "-" in token:
        return token.split("-", 1)[1]
    return token


def compare_vectors(gt_text: object, experiment_text: object) -> Tuple[int, float, str]:
    gt_tokens = normalize_vector(gt_text)
    experiment_tokens = normalize_vector(experiment_text)
    mismatches = 0
    highlighted_tokens: List[str] = []
    for predicate, gt_token, exp_token in zip(ACN_PREDICATES, gt_tokens, experiment_tokens):
        if gt_token == exp_token:
            highlighted_tokens.append(exp_token)
            continue
        mismatches += 1
        highlight = (
            f"{predicate}:[{_value_part(exp_token)}|GT={_value_part(gt_token)}]"
        )
        highlighted_tokens.append(highlight)
    total = len(ACN_PREDICATES)
    hamming = (total - mismatches) / total if total else 0.0
    highlighted_vector = " ".join(highlighted_tokens)
    return mismatches, hamming, highlighted_vector


def ensure_columns_exist(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in dataset: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    df = pd.read_excel(dataset_path)
    ensure_columns_exist(df, [args.ground_truth_column])
    ensure_columns_exist(df, args.experiments)

    gt_series = df[args.ground_truth_column]
    for experiment_column in args.experiments:
        diff_col = f"{experiment_column}_diff_count"
        hamming_col = f"{experiment_column}_hamming"
        highlight_col = f"{experiment_column}_highlighted"

        diffs: List[int] = []
        hammings: List[float] = []
        highlights: List[str] = []
        for gt_value, experiment_value in zip(gt_series, df[experiment_column]):
            diff, hamming, highlight = compare_vectors(gt_value, experiment_value)
            diffs.append(diff)
            hammings.append(round(hamming, 4))
            highlights.append(highlight)

        df[diff_col] = diffs
        df[hamming_col] = hammings
        df[highlight_col] = highlights

    output_path = args.output or dataset_path
    df.to_excel(output_path, index=False)
    print(
        f"Stored comparison columns for {len(args.experiments)} experiments in {output_path}"
    )


if __name__ == "__main__":
    main()
