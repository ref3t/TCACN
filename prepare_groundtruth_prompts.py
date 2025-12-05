#!/usr/bin/env python3
"""
Build chat-ready prompts for each labelled log entry in the GroundTruth dataset.

The script can either reuse the evaluate_llm.py template (without the taxonomy
block) or load a custom prompt from disk and inject the log in the placeholder
section (e.g. the Prompt.txt instructions shared with the LLM annotators).
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd

PROMPT_TEMPLATE_NO_TAX = """
You are a cybersecurity incident analyst trained to classify events using the Italian "ACN - Tassonomia Cyber". 
At the end of this prompt you will have LOGS/EVENTS about attack. Use ONLY the Italian "ACN - Tassonomia Cyber"(do not invent new categories) to classify and provide me the vector in one line.

------------------------------------------------------------

YOUR TASKS:

1. Parse the logs (IPs, ports, accounts, processes, timestamps, auth details, errors, anomalies).
2. Classify the event strictly according to the ACN taxonomy.
3. Output EXACTLY one value for each of the 17 predicates (strict order):
   1 BC:IM-<value>   2 BC:RO-<value>   3 BC:SE-<value>   4 TT:AC-<value>
   5 TT:AV-<value>   6 TT:FR-<value>   7 TT:BA-<value>   8 TT:DE-<value>
   9 TT:IG-<value>  10 TT:MA-<value>  11 TT:SO-<value>  12 TT:VU-<value>
  13 TA:AM-<value>  14 AC:IN-<value>  15 AC:OS-<value>  16 AC:PH-<value>
  17 AC:VE-<value>
   - Always output 1 value per predicate.
   - If no match exists, use NO/OT/UN as appropriate.

------------------------------------------------------------

The OUTPUT Response FORMAT:

One line: 17-field ACN vector (space-separated)

Example:
BC:IM-NO BC:RO-HU BC:SE-NO TT:AC-OT TT:AV-OT TT:FR-OT TT:BA-OT TT:DE-OT TT:IG-OT TT:MA-UN TT:SO-OT TT:VU-OT TA:AM-OT AC:IN-SR AC:OS-MI AC:PH-OT AC:VE-OT

------------------------------------------------------------

LOGS/EVENTS TO ANALYZE
{log_block}
"""


def wrap_texttt(text: str) -> str:
    safe = text.replace("\\", "\\\\").replace("}", "\\}")
    return f"\\texttt{{{safe}}}"


def build_log_block(row: pd.Series) -> str:
    """Return the evaluate_llm-style log block wrapped in LaTeX texttt."""
    log_text = textwrap.dedent(
        f"""
        _time: {_safe_field(row, '_time')}
        host: {_safe_field(row, 'host')}
        sourcetype: {_safe_field(row, 'sourcetype')}
        source: {_safe_field(row, 'source')}
        _raw:
        {_safe_field(row, '_raw', default='')}
        """
    ).strip()
    return wrap_texttt(log_text)


def build_prompt_without_taxonomy(log_block: str) -> str:
    return PROMPT_TEMPLATE_NO_TAX.format(log_block=log_block)


def _safe_field(row: pd.Series, key: str, default: str = "UNKNOWN") -> str:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return str(value)


def build_plain_log_block(row: pd.Series) -> str:
    """Return the log block exactly as shown in the Excel file (no LaTeX)."""
    log_text = textwrap.dedent(
        f"""
        _time: {_safe_field(row, '_time')}
        host: {_safe_field(row, 'host')}
        sourcetype: {_safe_field(row, 'sourcetype')}
        source: {_safe_field(row, 'source')}
        _raw:
        {_safe_field(row, '_raw', default='')}
        """
    ).strip()
    return log_text


def inject_log_into_template(
    template_text: str, log_block: str, placeholder: str
) -> str:
    """Replace the placeholder with the rendered log block."""
    if placeholder and placeholder in template_text:
        return template_text.replace(placeholder, log_block)
    appended = f"{template_text.rstrip()}\n{log_block}"
    return appended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach the evaluate_llm chat prompt (without the taxonomy section) "
            "to every labelled row in the GroundTruth spreadsheet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Data") / "GTAttacksLogs.xlsx",
        help="Path to the Excel file that holds the labelled events.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to store the updated Excel file. Defaults to --dataset "
            "and overwrites the source file when omitted."
        ),
    )
    parser.add_argument(
        "--prompt-column",
        default="prompt_no_taxonomy",
        help="Name of the column that will store the generated prompts.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "Optional plain-text prompt template. When set, the placeholder "
            "is replaced with the rendered log block from each row."
        ),
    )
    parser.add_argument(
        "--placeholder",
        default="<PUT LOGS HERE>",
        help=(
            "Marker to replace with the rendered log when --prompt-file is "
            "provided. The log is appended to the prompt when the marker is "
            "missing."
        ),
    )
    parser.add_argument(
        "--drop-unlabelled",
        action="store_true",
        help="Only keep rows where the TC-ACN column is not empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    output_path = args.output or dataset_path
    df = pd.read_excel(dataset_path)

    if args.drop_unlabelled and "TC-ACN" in df.columns:
        df = df[df["TC-ACN"].notna()].copy()

    if "_raw" not in df.columns:
        raise SystemExit("The dataset must contain an '_raw' column with the log text.")

    if args.prompt_column in df.columns:
        print(
            f"[info] Column {args.prompt_column} exists and will be overwritten."
        )

    prompt_template: str | None = None
    if args.prompt_file:
        if not args.prompt_file.exists():
            raise SystemExit(f"Prompt template not found: {args.prompt_file}")
        prompt_template = args.prompt_file.read_text(encoding="utf-8").strip()
        if args.placeholder and args.placeholder not in prompt_template:
            print(
                f"[warn] Placeholder '{args.placeholder}' not found in "
                f"{args.prompt_file}. The log block will be appended to the "
                "end of the prompt."
            )

    prompts = []
    for _, row in df.iterrows():
        if prompt_template:
            log_block = build_plain_log_block(row)
            prompt = inject_log_into_template(
                prompt_template, log_block, args.placeholder
            )
        else:
            log_block = build_log_block(row)
            prompt = build_prompt_without_taxonomy(log_block)
        prompts.append(prompt)

    df[args.prompt_column] = prompts
    df.to_excel(output_path, index=False)

    print(
        f"Stored {len(prompts)} prompts in column '{args.prompt_column}' "
        f"within {output_path}"
    )


if __name__ == "__main__":
    main()
