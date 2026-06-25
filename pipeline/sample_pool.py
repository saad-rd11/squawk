"""
Build the frozen CV pool: ~1000 ACNs stratified across aircraft, anomaly, year.

Usage:
    python -m pipeline.sample_pool --n 1000 --seed 42
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import DEFAULT_CONFIG


def _find_forced_acns(df: pd.DataFrame) -> set[int]:
    """Find ACNs matching deploy-known aircraft codes and exact-token narratives."""
    forced: set[int] = set()

    # Aircraft codes to force
    aircraft_patterns = {
        "B738": r"737[-\s]?800",
        "A320": r"\bA320\b",
        "C172": r"172",
        "B737": r"\bB737\b",
        "B767": r"\bB767\b",
        "CRJ9": r"\bCRJ[-\s]?9\b|Regional Jet 900",
        "E145": r"\bE145\b|EMB[-\s]?145",
    }
    col = "Make Model Name"
    n_per_type = 4
    for label, pattern in aircraft_patterns.items():
        matches = df[df[col].str.contains(pattern, na=False, regex=True)]
        if not matches.empty:
            for acn in matches.head(n_per_type)["ACN"]:
                forced.add(int(acn))

    # Exact-token narratives: engine codes, flight levels, TCAS events
    token_patterns = {
        "engine_stall": r"\bCFM56\b|\bPW4000\b|\bRB211\b|\bIAE V2500\b",
        "flight_level": r"\bFL\d{3}\b",
        "tcas_ra": r"\bTCAS\s+RA\b",
        "wind_shear": r"\bwindshear\b|\bwind shear\b",
        "bird_strike": r"\bbird[-\s]?strike\b|\bbird strike\b",
    }
    for label, pattern in token_patterns.items():
        matches = df[df["Narrative"].str.contains(pattern, na=False, regex=True)]
        if not matches.empty:
            for acn in matches.head(n_per_type)["ACN"]:
                forced.add(int(acn))

    return forced


def _anomaly_category(anomaly_str: str) -> str:
    """Bucket anomaly strings into coarse categories."""
    if pd.isna(anomaly_str) or not anomaly_str:
        return "none"
    s = anomaly_str.lower()
    if "nmac" in s or "midair" in s or "conflict" in s:
        return "conflict"
    if "cfit" in s or "terrain" in s:
        return "cfit"
    if "loss of control" in s or "loss of aircraft control" in s:
        return "loss_of_control"
    if "equipment" in s or "system" in s or "component" in s or "critical" in s:
        return "equipment"
    if "weather" in s or "turbulence" in s:
        return "weather"
    if "fuel" in s:
        return "fuel"
    if "bird" in s or "animal" in s:
        return "bird"
    if "atc" in s or "deviation" in s:
        return "atm"
    if "fire" in s or "smoke" in s or "fumes" in s:
        return "fire_smoke"
    return "other"


def _aircraft_category(make_model: str) -> str:
    """Bucket aircraft into coarse categories for stratification."""
    if pd.isna(make_model) or not make_model:
        return "unknown"
    s = make_model.lower()
    if (
        "helicopter" in s
        or "heli" in s
        or "rotor" in s
        or "bell" in s
        or "robinson" in s
        or "sikorsky" in s
    ):
        return "helicopter"
    if "uav" in s or "uas" in s or "drone" in s or "dji" in s or "unpiloted" in s:
        return "uas"
    if (
        "airliner" in s
        or "airbus" in s
        or "boeing" in s
        or "embraer" in s
        or "bombardier" in s
        or "737" in s
        or "747" in s
        or "757" in s
        or "767" in s
        or "777" in s
        or "787" in s
        or "a320" in s
        or "a330" in s
        or "a350" in s
        or "crj" in s
        or "erj" in s
        or "md-" in s
        or "dc-" in s
    ):
        return "airliner"
    if (
        "cessna" in s
        or "piper" in s
        or "beech" in s
        or "cirrus" in s
        or "mooney" in s
        or "diamond" in s
    ):
        return "general_aviation"
    if (
        "business" in s
        or "jet" in s
        or "gulfstream" in s
        or "citation" in s
        or "lear" in s
        or "falcon" in s
        or "challenger" in s
    ):
        return "business_jet"
    if "military" in s or "fighter" in s or "bomber" in s or "trainer" in s:
        return "military"
    if "glider" in s or "balloon" in s or "ultralight" in s:
        return "light_sport"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="cv_pool_acns.json")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    df = pd.read_csv(config.data_path, skiprows=1, low_memory=False)

    # Phase 1: forced ACNs (deploy-known codes + exact-token narratives)
    forced = _find_forced_acns(df)
    print(f"Forced ACNs: {len(forced)}")

    # Phase 2: annotate for stratification
    df["_anomaly_cat"] = df["Anomaly"].apply(_anomaly_category)
    df["_aircraft_cat"] = df["Make Model Name"].apply(_aircraft_category)
    df["_year"] = df["Date"].astype(str).str[:4].fillna("unknown")

    # Remove forced from candidate pool
    rng = np.random.RandomState(args.seed)
    forced_mask = df["ACN"].isin(forced)
    candidates = df[~forced_mask].copy()

    # Stratified sample: balance across anomaly category, aircraft category, year
    def _sample_stratum(group, n_per):
        return group.sample(n=min(n_per, len(group)), random_state=rng)

    samples_per_stratum = max(1, (args.n - len(forced)) // 12)
    sampled_parts = []

    for anomaly_cat in candidates["_anomaly_cat"].unique():
        for ac_cat in candidates["_aircraft_cat"].unique():
            sub = candidates[
                (candidates["_anomaly_cat"] == anomaly_cat)
                & (candidates["_aircraft_cat"] == ac_cat)
            ]
            if len(sub) == 0:
                continue
            # Further split by year
            for year in sub["_year"].unique():
                year_sub = sub[sub["_year"] == year]
                sampled = _sample_stratum(year_sub, max(1, samples_per_stratum // 3))
                sampled_parts.append(sampled)

    sampled = pd.concat(sampled_parts) if sampled_parts else pd.DataFrame()

    # Fill to target
    pool_acns = set(forced)
    for acn in sampled["ACN"]:
        if len(pool_acns) >= args.n:
            break
        pool_acns.add(int(acn))

    # If still under target, top off with random from remaining
    if len(pool_acns) < args.n:
        remaining = df[~df["ACN"].isin(pool_acns)]
        extra = remaining.sample(
            n=min(args.n - len(pool_acns), len(remaining)), random_state=rng
        )
        for acn in extra["ACN"]:
            pool_acns.add(int(acn))

    pool_acns_list = sorted(pool_acns)
    print(f"Pool size: {len(pool_acns_list)}")

    # Write
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(pool_acns_list, f)
    print(f"Wrote {output_path} ({len(pool_acns_list)} ACNs)")


if __name__ == "__main__":
    main()
