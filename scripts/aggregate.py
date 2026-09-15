"""Aggregate every summary parquet into one table.

Kept deliberately outside sim/: sweep workers must never write into a shared
output file, so aggregation is a separate, re-runnable step.

    python scripts/aggregate.py --out results/summary.parquet
"""
from __future__ import annotations

import argparse
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.logging_io import read_summaries  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/raw/*.parquet")
    ap.add_argument("--out", default="results/summary.parquet")
    ap.add_argument("--csv", default=None, help="also write a CSV copy")
    args = ap.parse_args()

    table = read_summaries(args.pattern)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pq.write_table(table, args.out, compression="snappy")
    print(f"{table.num_rows} runs, {table.num_columns} columns -> {args.out}")

    if args.csv:
        import pyarrow.csv as pv
        pv.write_csv(table, args.csv)
        print(f"csv -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
