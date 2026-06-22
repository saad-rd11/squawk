from typing import Any

import pandas as pd


def compute_aircraft_coverage(
    df: pd.DataFrame,
    generic_descriptors: set[str],
) -> dict[str, Any]:
    a1_raw = df["Make Model Name"].fillna("").str.strip()
    a1_null = df["Make Model Name"].isna()
    a1_nonnull = df["Make Model Name"].notna()
    is_generic = a1_raw.isin(generic_descriptors) & a1_nonnull
    is_partial_map = (
        a1_raw.str.contains(
            "Undifferentiated|Other Model|Next Generation", na=False, regex=True
        )
        & a1_nonnull
        & ~is_generic
    )
    resolved = a1_nonnull & ~is_generic & ~is_partial_map

    total = len(df)
    return {
        "total_rows": total,
        "resolved": int(resolved.sum()),
        "partial_map": int(is_partial_map.sum()),
        "generic_descriptor": int(is_generic.sum()),
        "null": int(a1_null.sum()),
        "sum_check": int(
            resolved.sum() + is_partial_map.sum() + is_generic.sum() + a1_null.sum()
        ),
    }


def compute_chunk_stats(df: pd.DataFrame) -> dict[str, Any]:
    chunks_per_row = df["narrative_chunks"].apply(len)
    total_chunks = chunks_per_row.sum()
    return {
        "total_chunks": int(total_chunks),
        "avg_chunks": round(chunks_per_row.mean(), 2),
        "pct_1_chunk": round((chunks_per_row == 1).mean() * 100, 1),
        "pct_1_to_2_chunks": round((chunks_per_row <= 2).mean() * 100, 1),
    }


def compute_all_metrics(
    df: pd.DataFrame,
    generic_descriptors: set[str],
) -> dict[str, Any]:
    return {
        "aircraft_coverage": compute_aircraft_coverage(df, generic_descriptors),
        "chunk_stats": compute_chunk_stats(df),
        "total_rows": len(df),
    }
