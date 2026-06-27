"""
Find real queries where sparse (BM25) beats dense (BGE), then test weighted RRF on them.

Strategy: probe with queries designed to trigger BM25's exact-match strength
(rare waypoints, specific technical terms, look-alike pairs) and identify
cases where sparse rank < dense rank for the gold ACN.
"""

import json
import logging
import statistics

import pandas as pd
from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 200
RRF_K = 60

CANDIDATE_QUERIES = [
    # Tier 1: Waypoint look-alikes (should strongly favor BM25)
    ("PONDD waypoint PONND navigation error similar names", 1852371),
    ("KENLN programmed instead of KENLL FMS fix entry error", 1942301),
    ("GOOFY intersection DSNEE arrival altitude crossing LGB", 1779610),
    ("KARLB2 arrival GOATZ COAZT similar sounding fixes", 1895732),
    ("BAYYY3 arrival HOU incomplete call sign deviation", 1721357),
    # Tier 2: Technical acronyms
    ("engine fire cartridge inadvertently discharged maintenance", 1807665),
    ("CFM56-7B fan blade replacement qualified personnel", 1852135),
    ("LAX SFRA NMAC exiting special flight rules area", 1857959),
    ("MVA violation SAT TRACON airborne conflict", 1896007),
    ("CFTT event night visual approach SJC", 1716462),
    ("VG actuating system screwdriver wire loom EMB-175", 1747114),
    ("Fan Exit Guide Vane Assembly damaged beyond repair", 1826087),
    ("NOTOC removed post flight B767", 1928526),
    # Tier 3: Rare terms
    ("spillout NMAC ZMA center controller", 1869827),
    ("MGTOW exceedance BE40", 1889906),
    ("TRUST certification flew over people recreational UAS", 1892901),
    ("autorotation NMAC helicopter practicing", 1902437),
    ("nordo R22 NMAC BH206", 1760769),
    # Additional strong candidates
    ("DRLLR FIVE STAR arrival IAH FMC late runway assignment", 1779057),
    ("DSD VOR TCAS RA training traffic", 1728376),
    ("EVB runway 7 taxiway E hold short lines faded", 1755782),
    ("false localizer capture lateral navigation mode", 1770840),
    ("Class 9 Hazmat liquid cleanup refusal policy", 1874854),
]


def _collapse_to_parents(scored_points):
    acn_scores = {}
    for sp in scored_points:
        rid = sp.payload.get("report_id")
        if rid is not None:
            acn = int(rid)
            score = sp.score
            if acn not in acn_scores or score > acn_scores[acn]:
                acn_scores[acn] = score
    return sorted(acn_scores.items(), key=lambda x: x[1], reverse=True)


def _parent_rank(collapsed, target_acn):
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn == target_acn:
            return rank
    return None


def rrf(dense, sparse, wd, ws):
    dr = {acn: rank for rank, (acn, _) in enumerate(dense, 1)}
    sr = {acn: rank for rank, (acn, _) in enumerate(sparse, 1)}
    all_acns = set(dr) | set(sr)
    default = max(len(dense), len(sparse)) + 1
    scores = {}
    for acn in all_acns:
        scores[acn] = wd / (RRF_K + dr.get(acn, default)) + ws / (
            RRF_K + sr.get(acn, default)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main():
    config = DEFAULT_CONFIG

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)

    with open("cv_pool_acns.json") as f:
        pool = set(json.load(f))

    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )

    query_texts = [q for q, _ in CANDIDATE_QUERIES]
    target_acns = [a for _, a in CANDIDATE_QUERIES]

    logger.info("Embedding %d queries...", len(query_texts))
    dvecs = ded.embed(query_texts)
    svecs = sed.embed(query_texts)

    mgr = Stage3Collection(path="./qdrant_storage")
    results = []

    for i, (query_text, target_acn) in enumerate(CANDIDATE_QUERIES):
        # Verify target is in pool
        if target_acn not in pool:
            results.append(
                {
                    "qnum": i + 1,
                    "query": query_text,
                    "target": target_acn,
                    "in_pool": False,
                    "d_rank": None,
                    "s_rank": None,
                }
            )
            continue

        dvec = dvecs[i]
        svec = svecs[i]

        dense_raw = mgr.client.query_points(
            COLLECTION_NAME,
            query=dvec,
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        dense_collapsed = _collapse_to_parents(dense_raw)

        sparse_raw = mgr.client.query_points(
            COLLECTION_NAME,
            query=SparseVector(indices=list(svec.keys()), values=list(svec.values())),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_collapsed = _collapse_to_parents(sparse_raw)

        old_combined = rrf(dense_collapsed, sparse_collapsed, 1.0, 1.0)
        new_combined = rrf(dense_collapsed, sparse_collapsed, 0.60, 0.40)

        d_rank = _parent_rank(dense_collapsed, target_acn)
        s_rank = _parent_rank(sparse_collapsed, target_acn)
        old_rank = _parent_rank(old_combined, target_acn)
        new_rank = _parent_rank(new_combined, target_acn)

        results.append(
            {
                "qnum": i + 1,
                "query": query_text,
                "target": target_acn,
                "in_pool": True,
                "d_rank": d_rank,
                "s_rank": s_rank,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "sparse_wins": s_rank is not None
                and (d_rank is None or s_rank < d_rank),
            }
        )

    mgr.close()

    # ================================================================
    # REPORT
    # ================================================================
    print("=" * 70)
    print("  SPARSE-WINS DETECTION: Candidate Queries")
    print("=" * 70)

    # Table header
    print(
        f"  {'Q':>3}  {'Target':>8}  {'Dense':>6}  {'Sparse':>6}  {'Old':>6}  {'New':>6}  {'Pool?':>5}  {'S>D?':>5}  {'Regr?':>6}"
    )
    print(
        f"  {'-' * 3}  {'-' * 8}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 5}  {'-' * 5}  {'-' * 6}"
    )

    sparse_wins_cases = []
    regressions = []

    for r in results:
        d_str = f"#{r['d_rank']}" if r["d_rank"] else "miss"
        s_str = f"#{r['s_rank']}" if r["s_rank"] else "miss"
        o_str = f"#{r['old_rank']}" if r.get("old_rank") else "miss"
        n_str = f"#{r['new_rank']}" if r.get("new_rank") else "miss"
        pool_str = "YES" if r["in_pool"] else "NO"
        sw_str = "YES" if r.get("sparse_wins") else "no"
        regr = ""
        if r.get("old_rank") and r.get("new_rank"):
            if r["new_rank"] > r["old_rank"]:
                regr = "REGR"
                regressions.append(r)
            elif r["new_rank"] < r["old_rank"]:
                regr = "IMPR"
        print(
            f"  {r['qnum']:>3}  {r['target']:>8}  {d_str:>6}  {s_str:>6}  {o_str:>6}  {n_str:>6}  {pool_str:>5}  {sw_str:>5}  {regr:>6}"
        )
        if r.get("sparse_wins"):
            sparse_wins_cases.append(r)

    print()
    print("=" * 70)
    print(f"  SPARSE-WINS QUERIES ({len(sparse_wins_cases)} found)")
    print("=" * 70)

    for r in sparse_wins_cases:
        d_str = f"#{r['d_rank']}" if r["d_rank"] else "miss"
        s_str = f"#{r['s_rank']}" if r["s_rank"] else "miss"
        old_str = f"#{r['old_rank']}" if r["old_rank"] else "miss"
        new_str = f"#{r['new_rank']}" if r["new_rank"] else "miss"
        synopsis = (
            str(df[df["ACN"] == r["target"]].iloc[0]["Synopsis"])[:120]
            if r["target"] in df["ACN"].values
            else "?"
        )
        print(f"\n  Q{r['qnum']}: {r['query']}")
        print(
            f"    Target ACN {r['target']}: dense={d_str}, sparse={s_str}, old_RRF={old_str}, new_RRF={new_str}"
        )
        print(f"    Synopsis: {synopsis}")

    # Check regressions specifically on sparse-wins cases
    print()
    print("=" * 70)
    print("  REGRESSION ANALYSIS ON SPARSE-WINS CASES")
    print("=" * 70)
    sw_regressed = [
        r
        for r in sparse_wins_cases
        if r["new_rank"] and r["old_rank"] and r["new_rank"] > r["old_rank"]
    ]
    if sw_regressed:
        print(f"  {len(sw_regressed)} sparse-wins queries regressed with weighted RRF:")
        for r in sw_regressed:
            print(f"    Q{r['qnum']}: old=#{r['old_rank']} → new=#{r['new_rank']}")
            print(f"      {r['query']}")
    else:
        print(
            f"  Zero regressions on sparse-wins queries. Weighted RRF preserved all sparse benefits."
        )

    # Detailed per-sparse-wins analysis
    print()
    print("=" * 70)
    print("  DETAILED ANALYSIS")
    print("=" * 70)
    for r in sorted(sparse_wins_cases, key=lambda x: x["s_rank"]):
        d_str = f"#{r['d_rank']}" if r["d_rank"] else "miss"
        s_str = f"#{r['s_rank']}" if r["s_rank"] else "miss"
        old_str = f"#{r['old_rank']}" if r["old_rank"] else "miss"
        new_str = f"#{r['new_rank']}" if r["new_rank"] else "miss"
        verdict = (
            "✓ PRESERVED"
            if (old_str == new_str)
            else (
                f"⚠ IMPROVED"
                if (r["new_rank"] and r["old_rank"] and r["new_rank"] < r["old_rank"])
                else f"✗ REGRESSED"
            )
        )
        print(
            f"  Q{r['qnum']:>2}: dense={d_str} sparse={s_str} → old_RRF={old_str} new_RRF={new_str}  {verdict}"
        )


if __name__ == "__main__":
    main()
