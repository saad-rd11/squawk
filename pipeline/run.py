"""
Production pipeline: load, normalize, validate, output.

Usage:
    python -m pipeline.run [--config path/to/config.py]
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .metrics import compute_all_metrics
from .normalize import (
    NormalizationCounters,
    load_aircraft_map,
    load_generic_descriptors,
    normalize_text,
)
from .transform import chunk_text, transform_row
from .validate import validate_aircraft_map, validate_points

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1, low_memory=False)
    if "Unnamed: 125" in df.columns and df["Unnamed: 125"].isna().all():
        df = df.drop(columns=["Unnamed: 125"])
    acn_dupes = df["ACN"].duplicated(keep="first")
    if acn_dupes.any():
        logger.warning("Dropped %d duplicate ACNs", acn_dupes.sum())
        df = df[~acn_dupes].copy()
    return df


def run(config: PipelineConfig = DEFAULT_CONFIG) -> dict:
    logger.info("Loading data from %s", config.data_path)
    df = load_data(config.data_path)

    aircraft_map = load_aircraft_map(config.aircraft_map_path)
    generic_descriptors = load_generic_descriptors(config.generic_descriptors_path)
    logger.info(
        "Loaded aircraft_map (%d entries), generic_descriptors (%d entries)",
        len(aircraft_map),
        len(generic_descriptors),
    )

    map_validation = validate_aircraft_map(aircraft_map, config.icao_designators_path)
    if map_validation["status"] == "fail":
        for err in map_validation["errors"]:
            logger.error("Aircraft map validation: %s", err)
        raise ValueError("Aircraft map validation failed — fix errors before ingesting")
    logger.info(
        "Aircraft map validation passed (%d entries)", map_validation["total_entries"]
    )

    from anomaly_map import expand_anomalies, validate_export  # noqa: E402

    validate_export(df["Anomaly"])
    logger.info("validate_export passed (100%% anomaly coverage)")

    rows_before = len(df)
    df = df[df["Narrative"].notna()].copy()
    if len(df) < rows_before:
        logger.info("Dropped %d rows with no Narrative 1", rows_before - len(df))
    rows_before = len(df)
    df = df[df["Synopsis"].notna()].copy()
    if len(df) < rows_before:
        logger.info("Dropped %d rows with no Synopsis", rows_before - len(df))

    parent_points = []
    child_points = []
    counters = NormalizationCounters()

    # Intentional: no row-level isolation — fail fast on bad data.
    # A/B experiments validate upstream; skipping would silently mask corrupt rows.
    for idx, row in df.iterrows():
        pp, cps = transform_row(
            row.to_dict(), aircraft_map, config, expand_anomalies, counter=counters
        )
        parent_points.append(pp)
        child_points.extend(cps)

    norm_report = counters.report()
    logger.info("Normalization counters: %s", norm_report)

    # Intentional: separate chunk pass for metrics — same transform, independent concern.
    # Sharing chunks from transform_row would couple measurement to pipeline internals.
    # 2x string cost on in-memory data is negligible.
    # Normalize narratives for metrics
    df["narrative_clean"] = df["Narrative"].apply(normalize_text)

    # Chunk for metrics
    df["narrative_chunks"] = df["narrative_clean"].apply(
        lambda t: chunk_text(t, max_chars=config.chunk_max_chars)
    )

    # Chunk stats
    chunks_per_row = df["narrative_chunks"].apply(len)
    total_chunks = chunks_per_row.sum()
    logger.info(
        "Chunk stats: %d total, %.2f avg, %.1f%% 1-chunk, %.1f%% 1-2 chunks",
        total_chunks,
        chunks_per_row.mean(),
        (chunks_per_row == 1).mean() * 100,
        (chunks_per_row <= 2).mean() * 100,
    )
    logger.info(
        "Points: %d parent, %d child, %d total",
        len(parent_points),
        len(child_points),
        len(parent_points) + len(child_points),
    )

    validation_errors = validate_points(parent_points, child_points)
    if validation_errors == 0:
        logger.info("Pydantic validation passed on all points")
    else:
        logger.error("%d validation errors found", validation_errors)

    total = parent_points + child_points
    total.sort(key=lambda p: p.id)

    output_path = Path(config.output_path)
    with open(output_path, "w") as f:
        for p in total:
            f.write(p.model_dump_json() + "\n")
    logger.info(
        "Wrote %s (%d points, %d bytes)",
        output_path,
        len(total),
        os.path.getsize(output_path),
    )

    metrics = compute_all_metrics(df, generic_descriptors)
    metrics["normalization"] = counters.report()
    return metrics


if __name__ == "__main__":
    metrics = run()
    print()
    print("=" * 60)
    print("METRICS")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
