"""
Quick cos-sim survey: which ASRS fields have high standalone query similarity?

Tests a representative sample of query->field cos sims to identify signals
that could benefit from being separate child points (like the anomaly child).

Usage:
    ./venv/bin/python -m pipeline.signal_survey
"""

import json
import logging
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

QUERY_SAMPLE = [
    61,
    70,
    38,
    71,
    53,
    66,
    48,
    51,
    52,
    37,
    16,
    25,
    34,
    55,
    29,
]

FIELD_TEMPLATES = {
    "Synopsis": "Synopsis: {}",
    "Primary Problem": "Primary Problem: {}",
    "Contributing Factors": "Contributing Factors: {}",
    "Flight Phase": "Flight Phase: {}",
    "Component": "Component: {}",
    "Flight Conditions": "Flight Conditions: {}",
    "Flight Plan": "Flight Plan: {}",
    "Narrative.1 (2nd pilot)": "Narrative: {}",
    "Raw Anomaly (unmapped)": "{}",
    "Narrative (baseline)": "Narrative: {}",
}


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_leakage(query_text, narrative_text):
    qt = set(query_text.lower().split())
    nt = set(str(narrative_text).lower().split())
    if not qt:
        return 0.0
    return len(qt & nt) / len(qt)


def main():
    with open("eval_queries.json") as f:
        all_queries_list = json.load(f)
    all_queries = {i + 1: q for i, q in enumerate(all_queries_list)}

    query_items = [
        (qnum, all_queries[qnum]) for qnum in QUERY_SAMPLE if qnum in all_queries
    ]

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)

    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))
    pool_df = df[df["ACN"].isin(pool_acns)].set_index("ACN")

    query_fields = []
    for qnum, q in query_items:
        for acn in q["expected_acns"]:
            if acn not in pool_df.index:
                continue
            row = pool_df.loc[acn]
            fields = {
                "Synopsis": str(row.get("Synopsis", "")),
                "Primary Problem": str(row.get("Primary Problem", "")),
                "Contributing Factors": str(
                    row.get("Contributing Factors / Situations", "")
                ),
                "Flight Phase": str(row.get("Flight Phase", "")),
                "Component": str(row.get("Aircraft Component", "")),
                "Flight Conditions": str(row.get("Flight Conditions", "")),
                "Flight Plan": str(row.get("Flight Plan", "")),
                "Narrative.1 (2nd pilot)": str(row.get("Narrative.1", "")),
                "Raw Anomaly (unmapped)": str(row.get("Anomaly", "")),
                "Narrative (baseline)": str(row.get("Narrative", "")),
            }
            leakage = compute_leakage(q["query"], fields["Narrative (baseline)"])
            query_fields.append(
                {
                    "qnum": qnum,
                    "acn": acn,
                    "query": q["query"],
                    "fields": fields,
                    "leakage": leakage,
                    "qtype": q.get("type", "conceptual"),
                }
            )

    logger.info("Query-ACN pairs: %d", len(query_fields))

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    all_field_texts = []
    metadata = []
    for qf_idx, qf in enumerate(query_fields):
        for fname, template in FIELD_TEMPLATES.items():
            raw_val = qf["fields"][fname]
            if not raw_val or raw_val.lower() in (
                "nan",
                "",
                "none",
                "nat",
                "no specific anomaly occurred unwanted situation",
                "no specific anomaly occurred all types",
            ):
                continue
            text = template.format(raw_val[:2000])
            if len(text.strip()) < 3:
                continue
            all_field_texts.append(text)
            metadata.append((qf_idx, fname))

    logger.info("Embedding %d field texts...", len(all_field_texts))
    embeddings = model.encode(all_field_texts, show_progress_bar=True)

    query_texts = [qf["query"] for qf in query_fields]
    query_embs = model.encode(query_texts, show_progress_bar=True)

    results = []
    for emb_idx, (qf_idx, fname) in enumerate(metadata):
        sim = cos_sim(query_embs[qf_idx], embeddings[emb_idx])
        qf = query_fields[qf_idx]
        results.append(
            {
                "qnum": qf["qnum"],
                "acn": qf["acn"],
                "field": fname,
                "cos_sim": sim,
                "leakage": qf["leakage"],
                "query": qf["query"],
                "value": all_field_texts[emb_idx][:80],
            }
        )

    # Aggregate per field
    field_sims = defaultdict(list)
    for r in results:
        field_sims[r["field"]].append(r["cos_sim"])

    LEAKAGE_THRESHOLD = 0.353

    print("\n" + "=" * 75)
    print(
        "  SIGNAL SURVEY — Cos-sim by field (pool ACNs, %d q-acn pairs)"
        % len(query_fields)
    )
    print("=" * 75)

    print(f"\n{'Field':<27} {'N':>4} {'Mean':>7} {'Median':>7} {'Min':>7} {'Max':>7}")
    print("-" * 62)
    for fname in sorted(
        field_sims, key=lambda f: statistics.mean(field_sims[f]), reverse=True
    ):
        vals = field_sims[fname]
        print(
            f"{fname:<27} {len(vals):>4} {statistics.mean(vals):>7.4f} {statistics.median(vals):>7.4f} {min(vals):>7.4f} {max(vals):>7.4f}"
        )

    # Per-query detail (low-leakage only unless specified)
    print("\n\n" + "=" * 75)
    print("  PER-QUERY DETAIL — sorted by cos sim descending")
    print("=" * 75)

    by_qnum = defaultdict(list)
    for r in results:
        by_qnum[r["qnum"]].append(r)

    for qnum in sorted(by_qnum):
        rows = by_qnum[qnum]
        leakage = rows[0]["leakage"]
        print(f'\nQ{qnum:>2} (leak={100 * leakage:.0f}%) "{rows[0]["query"][:55]}"')
        print(f"{'Field':<27} {'Cos Sim':>7}  Value")
        print("-" * 95)
        for r in sorted(rows, key=lambda x: x["cos_sim"], reverse=True):
            print(f"{r['field']:<27} {r['cos_sim']:>7.4f}  {r['value'][:58]}")

    # Summary
    print("\n\n" + "=" * 75)
    print("  SUMMARY: Δ vs baseline narrative (positive = worth investigating)")
    print("=" * 75)
    baseline_mean = statistics.mean(field_sims.get("Narrative (baseline)", [0]))
    baseline_med = statistics.median(field_sims.get("Narrative (baseline)", [0]))

    for fname in sorted(
        field_sims, key=lambda f: statistics.mean(field_sims[f]), reverse=True
    ):
        if fname == "Narrative (baseline)":
            continue
        f_mean = statistics.mean(field_sims[fname])
        f_med = statistics.median(field_sims[fname])
        d_mean = f_mean - baseline_mean
        d_med = f_med - baseline_med

        if d_mean > 0.05:
            marker = " ★ HIGH"
        elif d_mean > 0.0:
            marker = " ◆ MODEST"
        else:
            marker = "   FLAT/NEG"

        print(
            f"  {marker}  {fname:<27} mean={f_mean:.4f} (Δ={d_mean:+.4f})  median={f_med:.4f} (Δ={d_med:+.4f})"
        )

    print(
        f"\n  Baseline (Narrative): mean={baseline_mean:.4f}  median={baseline_med:.4f}"
    )
    print(
        f"\n  Raw Anomaly mean={statistics.mean(field_sims.get('Raw Anomaly (unmapped)', [0])):.4f} — known win, validates method"
    )


if __name__ == "__main__":
    main()
