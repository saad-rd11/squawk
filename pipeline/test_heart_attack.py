"""
Test that dense-weighted RRF fixes the heart-attack→battery fusion artifact.
Compares old equal-weight RRF vs new dense-weighted RRF for Q13.
"""

import json
import logging

import pandas as pd
from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 100
RRF_K = 60
W_DENSE = 0.60
W_SPARSE = 0.40

QUERY = "passenger having a heart attack mid-flight paramedics met at the gate"

# Expected relevant ACNs for sick passenger
SICK_PASSENGER_ACN = (
    1766047  # "Flight attendants reported dealing with a sick passenger during flight"
)
BATTERY_ACN = 1795879  # "B737 flight crew reported being notified... spare lithium-ion battery..."


def collapse(scored_points):
    out = {}
    for sp in scored_points:
        rid = sp.payload.get("report_id")
        if rid is not None:
            a = int(rid)
            if a not in out or sp.score > out[a]:
                out[a] = sp.score
    return sorted(out.items(), key=lambda x: x[1], reverse=True)


def rrf_equal(dense, sparse):
    dm = {a: r for r, (a, _) in enumerate(dense, 1)}
    sm = {a: r for r, (a, _) in enumerate(sparse, 1)}
    default = max(len(dm), len(sm)) + 1
    scores = {}
    for a in set(dm) | set(sm):
        scores[a] = 1.0 / (RRF_K + dm.get(a, default)) + 1.0 / (
            RRF_K + sm.get(a, default)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rrf_weighted(dense, sparse, wd=W_DENSE, ws=W_SPARSE):
    dm = {a: r for r, (a, _) in enumerate(dense, 1)}
    sm = {a: r for r, (a, _) in enumerate(sparse, 1)}
    default = max(len(dm), len(sm)) + 1
    scores = {}
    for a in set(dm) | set(sm):
        scores[a] = wd / (RRF_K + dm.get(a, default)) + ws / (
            RRF_K + sm.get(a, default)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def find_rank(collapsed, target_acn):
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn == target_acn:
            return rank
    return None


def print_top_5(collapsed, label, df, highlight=None):
    print(f"\n  --- {label} ---")
    print(f"  {'Rank':>4}  {'ACN':>8}  {'Score':>10}  {'Synopsis'}")
    for rank, (acn, score) in enumerate(collapsed[:5], 1):
        row = df[df["ACN"] == acn]
        syn = str(row.iloc[0].get("Synopsis", ""))[:100] if not row.empty else "?"
        marker = (
            " <<< SICK"
            if acn == highlight
            else (" BATTERY" if acn == BATTERY_ACN else "")
        )
        print(f"  {rank:>4}  {acn:>8}  {score:>10.4f}  {syn}{marker}")


def main():
    config = DEFAULT_CONFIG

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)

    sick_row = df[df["ACN"] == SICK_PASSENGER_ACN]
    bat_row = df[df["ACN"] == BATTERY_ACN]

    print("=" * 70)
    print("  HEART ATTACK QUERY: DENSE-WEIGHTED RRF VERIFICATION")
    print("=" * 70)
    print(f'\n  Query: "{QUERY}"')
    print(f"\n  Target report: ACN {SICK_PASSENGER_ACN}")
    print(f"    Synopsis: {sick_row.iloc[0]['Synopsis']}")
    print(f"    In pool: {SICK_PASSENGER_ACN in set()}")
    print(f"\n  Spurious report: ACN {BATTERY_ACN}")
    print(f"    Synopsis: {bat_row.iloc[0]['Synopsis'][:100]}...")

    with open("cv_pool_acns.json") as f:
        pool = set(json.load(f))

    print(f"  Sick passenger in pool: {SICK_PASSENGER_ACN in pool}")
    print(f"  Battery report in pool: {BATTERY_ACN in pool}")

    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )

    dvec = ded.embed([QUERY])[0]
    svec = sed.embed([QUERY])[0]

    mgr = Stage3Collection(path="./qdrant_storage")

    dense_raw = mgr.client.query_points(
        COLLECTION_NAME,
        query=dvec,
        using="dense",
        limit=SEARCH_LIMIT,
        with_payload=["report_id", "narrative_source"],
    ).points
    dense_collapsed = collapse(dense_raw)

    sparse_raw = mgr.client.query_points(
        COLLECTION_NAME,
        query=SparseVector(indices=list(svec.keys()), values=list(svec.values())),
        using="sparse",
        limit=SEARCH_LIMIT,
        with_payload=["report_id"],
    ).points
    sparse_collapsed = collapse(sparse_raw)

    mgr.close()

    print_top_5(dense_collapsed, "DENSE ONLY", df, highlight=SICK_PASSENGER_ACN)
    print_top_5(sparse_collapsed, "SPARSE ONLY", df, highlight=SICK_PASSENGER_ACN)

    equal_combined = rrf_equal(dense_collapsed, sparse_collapsed)
    weighted_combined = rrf_weighted(dense_collapsed, sparse_collapsed)

    print_top_5(
        equal_combined,
        "EQUAL-WEIGHT RRF (old, w_d=1.0, w_s=1.0)",
        df,
        highlight=SICK_PASSENGER_ACN,
    )
    print_top_5(
        weighted_combined,
        f"DENSE-WEIGHTED RRF (new, w_d={W_DENSE:.1f}, w_s={W_SPARSE:.1f})",
        df,
        highlight=SICK_PASSENGER_ACN,
    )

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for method, collapsed in [
        ("Dense only", dense_collapsed),
        ("Sparse only", sparse_collapsed),
        ("Equal-weight RRF", equal_combined),
        ("Dense-weighted RRF", weighted_combined),
    ]:
        sr = find_rank(collapsed, SICK_PASSENGER_ACN)
        br = find_rank(collapsed, BATTERY_ACN)
        sick_wins = sr is not None and (br is None or sr < br)
        print(
            f"  {method:25s}: sick_passenger=#{sr or 'miss':<5} battery=#{br or 'miss':<5}  {'✓ SICK WINS' if sick_wins else '✗ BATTERY WINS'}"
        )

    # Also check if there are any sick passenger ACNs in the pool
    sick_acns = df[
        df["Synopsis"]
        .str.lower()
        .str.contains(
            "sick passenger|medical.*emergency|passenger.*ill|passenger.*paramedic",
            na=False,
        )
    ]["ACN"].tolist()
    print(
        f"\n  All sick-passenger-related ACNs in pool: {[a for a in sick_acns if a in pool]}"
    )
    print(f"  All sick-passenger-related ACNs overall: {sick_acns}")


if __name__ == "__main__":
    main()
