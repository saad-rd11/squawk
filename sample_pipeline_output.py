"""
Sample rows from ASRS data, run through pipeline, write CSV with
full original text side-by-side with pipeline-normalized output.

Usage:
    python3 sample_pipeline_output.py [--seed 42] [--n 5]
"""

import argparse
import csv

import pandas as pd

from pipeline.config import DEFAULT_CONFIG
from pipeline.normalize import load_aircraft_map, normalize_text
from pipeline.transform import transform_row
from anomaly_map import expand_anomalies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output", default="pipeline_sample.csv")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    aircraft_map = load_aircraft_map(config.aircraft_map_path)

    df = pd.read_csv(config.data_path, skiprows=1, low_memory=False)
    sample = df.sample(n=args.n, random_state=args.seed)

    rows_out = []
    for _, row in sample.iterrows():
        rd = row.to_dict()
        pp, cps = transform_row(rd, aircraft_map, config, expand_anomalies)

        rows_out.append(
            {
                "acn": rd["ACN"],
                "original_make_model": rd.get("Make Model Name", ""),
                "original_anomaly": rd.get("Anomaly", ""),
                "original_flight_phase": rd.get("Flight Phase", ""),
                "original_flight_conditions": rd.get("Flight Conditions", ""),
                "original_flight_plan": rd.get("Flight Plan", ""),
                "original_state": rd.get("State Reference", ""),
                "original_operator": rd.get("Aircraft Operator", ""),
                "original_synopsis": rd.get("Synopsis", ""),
                "original_narrative": rd.get("Narrative", ""),
                "pipeline_aircraft_models": "; ".join(pp.payload.aircraft_models),
                "pipeline_aircraft_family": pp.payload.aircraft_family,
                "pipeline_manufacturer": pp.payload.manufacturer,
                "pipeline_flight_phase": "; ".join(pp.payload.flight_phase),
                "pipeline_anomaly_codes": "; ".join(pp.payload.anomaly),
                "pipeline_flight_conditions": pp.payload.flight_conditions or "",
                "pipeline_flight_plan": pp.payload.flight_plan or "",
                "pipeline_state": pp.payload.state or "",
                "pipeline_operator": pp.payload.operator or "",
                "pipeline_year": pp.payload.year or "",
                "pipeline_synopsis": normalize_text(rd.get("Synopsis", "")),
                "pipeline_narrative": normalize_text(rd.get("Narrative", "")),
                "parent_id": pp.id,
                "num_chunks": len(cps),
                "context_prefix": cps[0].payload.context_prefix if cps else "",
            }
        )

    fieldnames = [
        "acn",
        "original_make_model",
        "original_anomaly",
        "original_flight_phase",
        "original_flight_conditions",
        "original_flight_plan",
        "original_state",
        "original_operator",
        "original_synopsis",
        "original_narrative",
        "pipeline_aircraft_models",
        "pipeline_aircraft_family",
        "pipeline_manufacturer",
        "pipeline_flight_phase",
        "pipeline_anomaly_codes",
        "pipeline_flight_conditions",
        "pipeline_flight_plan",
        "pipeline_state",
        "pipeline_operator",
        "pipeline_year",
        "pipeline_synopsis",
        "pipeline_narrative",
        "parent_id",
        "num_chunks",
        "context_prefix",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {args.output}")
    print(f"Columns: {', '.join(fieldnames)}")


if __name__ == "__main__":
    main()
