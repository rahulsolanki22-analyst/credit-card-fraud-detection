import argparse
import os

import numpy as np
import pandas as pd


def add_time_column(
    input_csv: str,
    output_csv: str,
    *,
    time_step: float = 1.0,
    time_start: float = 0.0,
    chunksize: int = 200_000,
    use_id_if_present: bool = True,
    insert_after_id: bool = True,
) -> None:
    """
    Add a `Time` column to a large CSV in a memory-safe way (chunked streaming).

    - If `use_id_if_present` and the CSV has an `id` column, Time is computed as:
        Time = time_start + id * time_step
    - Otherwise Time is computed from row order:
        Time = time_start + row_index * time_step
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    reader = pd.read_csv(input_csv, chunksize=chunksize)
    first = True
    offset = 0

    for chunk in reader:
        if "Time" in chunk.columns:
            raise ValueError("Input CSV already contains a `Time` column.")

        if use_id_if_present and "id" in chunk.columns:
            time_values = time_start + chunk["id"].astype(float) * time_step
            insert_at = 1 if insert_after_id else 0
        else:
            time_values = time_start + (np.arange(len(chunk)) + offset) * time_step
            insert_at = 0

        chunk.insert(insert_at, "Time", time_values)

        chunk.to_csv(
            output_csv,
            mode="w" if first else "a",
            header=first,
            index=False,
        )

        first = False
        offset += len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a `Time` column to a large CSV (streaming/chunked)."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV (e.g. creditcard_2023.csv)")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV (e.g. creditcard_2023_with_time.csv)",
    )
    parser.add_argument("--time-step", type=float, default=1.0, help="Time increment per row/id.")
    parser.add_argument("--time-start", type=float, default=0.0, help="Starting Time value.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk.")
    parser.add_argument(
        "--no-use-id",
        action="store_true",
        help="Do not derive Time from `id` even if `id` exists; use row order instead.",
    )
    args = parser.parse_args()

    add_time_column(
        args.input,
        args.output,
        time_step=args.time_step,
        time_start=args.time_start,
        chunksize=args.chunksize,
        use_id_if_present=not args.no_use_id,
    )

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

