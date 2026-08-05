#!/usr/bin/env python3
"""
Filter landscape rows from a video stats CSV using only CSV columns.
"""

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT_CSV = (
    "/m2v_intern/xuyifan09/projects/zjm_video_seg/data/"
    "KlingVA-0312_10w_1080p/video_stats.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter landscape rows from a video stats CSV."
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to the filtered CSV. Defaults to '<input>_landscape.csv'.",
    )
    parser.add_argument(
        "--width-column",
        default="width",
        help="Column name for video width.",
    )
    parser.add_argument(
        "--height-column",
        default="height",
        help="Column name for video height.",
    )
    return parser.parse_args()


def get_default_output_path(input_csv):
    input_path = Path(input_csv)
    return str(input_path.with_name(f"{input_path.stem}_landscape{input_path.suffix}"))


def parse_dimension(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def main():
    args = parse_args()
    output_csv = args.output_csv or get_default_output_path(args.input_csv)

    with open(args.input_csv, "r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"No header found in CSV: {args.input_csv}")

    if args.width_column not in fieldnames:
        raise ValueError(f"Missing width column: {args.width_column}")
    if args.height_column not in fieldnames:
        raise ValueError(f"Missing height column: {args.height_column}")

    total_rows = len(rows)
    invalid_rows = 0
    landscape_rows = 0

    with open(output_csv, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            width = parse_dimension(row.get(args.width_column))
            height = parse_dimension(row.get(args.height_column))

            if width is None or height is None:
                invalid_rows += 1
                continue

            if width > height:
                writer.writerow(row)
                landscape_rows += 1

    print(f"input_csv={args.input_csv}")
    print(f"output_csv={output_csv}")
    print(f"total_rows={total_rows}")
    print(f"landscape_rows={landscape_rows}")
    print(f"invalid_rows={invalid_rows}")


if __name__ == "__main__":
    main()
