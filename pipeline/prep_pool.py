"""
Filter pipeline output to the CV pool and write a CSV with raw + pipeline fields,
nothing truncated. This is the human tool for verifying Gemini proposals and
hand-building exact-token queries.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pipeline.config import DEFAULT_CONFIG


def _extract_point_info(pt: dict) -> dict:
    """Extract ACN and flattened payload from a pipeline point."""
    pt_id = pt["id"]
    if "_n" in pt_id and pt_id.rsplit("_n", 1)[1].isdigit():
        base = pt_id.rsplit("_n", 1)[0]
    else:
        base = pt_id
    acn = int(base.replace("asrs_", ""))
    flat = {
        "acn": acn,
        "id": pt_id,
        "point_type": "child" if "sparse" in pt else "parent",
    }
    flat["vector_model"] = pt.get("vector", "")
    flat["sparse_model"] = pt.get("sparse", "")
    payload = pt.get("payload", {})
    for k, v in payload.items():
        if isinstance(v, list):
            flat[f"payload.{k}"] = "; ".join(str(x) for x in v)
        elif v is None:
            flat[f"payload.{k}"] = ""
        else:
            flat[f"payload.{k}"] = v
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="cv_pool_acns.json")
    parser.add_argument("--points", default="points.jsonl")
    parser.add_argument("--output", default="pool_full_output.csv")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    csv_path = args.csv or config.data_path

    with open(args.pool) as f:
        pool_acns = set(json.load(f))
    print(f"Pool: {len(pool_acns)} ACNs")

    # Load raw CSV, filter to pool
    raw_df = pd.read_csv(csv_path, skiprows=1, low_memory=False)
    raw_df = raw_df[raw_df["ACN"].isin(pool_acns)].copy()
    raw_df.rename(columns={"ACN": "acn"}, inplace=True)
    print(f"Raw pool rows: {len(raw_df)}")

    # Load pipeline output JSONL, filter to pool
    parent_rows = []
    for line in open(args.points):
        pt = json.loads(line)
        info = _extract_point_info(pt)
        if info["acn"] not in pool_acns:
            continue
        if info["point_type"] == "parent":
            parent_rows.append(info)

    pipe_df = pd.DataFrame(parent_rows)
    print(f"Pipeline pool parents: {len(pipe_df)}")

    # Merge raw + pipeline parent data on acn
    merged = raw_df.merge(pipe_df, on="acn", how="inner", suffixes=("_raw", ""))
    print(f"Merged rows: {len(merged)}, columns: {len(merged.columns)}")

    # Write — every column, no truncation
    output_path = Path(args.output)
    merged.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(merged)} rows, {len(merged.columns)} cols)")


if __name__ == "__main__":
    main()
