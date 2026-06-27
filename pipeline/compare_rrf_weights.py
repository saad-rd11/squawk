"""
Per-query comparison: old equal-weight RRF vs new dense-weighted RRF (w_d=0.80, w_s=0.20).

Identifies:
- Queries where sparse saved dense (dense missed, sparse found it)
- Whether weighted RRF regressed any sparse-saved queries
- Per-query delta between old and new
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
OLD_WD, OLD_WS = 1.0, 1.0
NEW_WD, NEW_WS = 0.80, 0.20


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


def _parent_mrr(collapsed, expected):
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn in expected:
            return 1.0 / rank
    return 0.0


def _parent_rank(collapsed, expected):
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn in expected:
            return rank
    return None


def _parent_found(collapsed, expected):
    return any(acn in expected for acn, _ in collapsed)


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

    with open("eval_queries.json") as f:
        queries = json.load(f)

    n = len(queries)

    dense_embedder = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sparse_embedder = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )

    query_texts = [q["query"] for q in queries]
    logger.info("Embedding queries...")
    dense_vecs = dense_embedder.embed(query_texts)
    sparse_vecs = sparse_embedder.embed(query_texts)

    mgr = Stage3Collection(path="./qdrant_storage")
    info = mgr.client.get_collection(COLLECTION_NAME)
    logger.info("Collection: %d points", info.points_count)

    # Per-query results
    rows = []
    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        dvec = dense_vecs[i]
        svec = sparse_vecs[i]
        qtype = q.get("type", "conceptual")

        dense_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dvec,
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        dense_collapsed = _collapse_to_parents(dense_raw)

        sparse_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(indices=list(svec.keys()), values=list(svec.values())),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_collapsed = _collapse_to_parents(sparse_raw)

        old_combined = rrf(dense_collapsed, sparse_collapsed, OLD_WD, OLD_WS)
        new_combined = rrf(dense_collapsed, sparse_collapsed, NEW_WD, NEW_WS)

        d_mrr = _parent_mrr(dense_collapsed, expected)
        s_mrr = _parent_mrr(sparse_collapsed, expected)
        old_mrr = _parent_mrr(old_combined, expected)
        new_mrr = _parent_mrr(new_combined, expected)

        d_rank = _parent_rank(dense_collapsed, expected)
        s_rank = _parent_rank(sparse_collapsed, expected)
        old_rank = _parent_rank(old_combined, expected)
        new_rank = _parent_rank(new_combined, expected)

        d_found = _parent_found(dense_collapsed, expected)
        s_found = _parent_found(sparse_collapsed, expected)

        rows.append(
            {
                "qnum": i + 1,
                "qtype": qtype,
                "query": q["query"],
                "expected": expected,
                "d_found": d_found,
                "s_found": s_found,
                "d_rank": d_rank,
                "s_rank": s_rank,
                "d_mrr": d_mrr,
                "s_mrr": s_mrr,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "old_mrr": old_mrr,
                "new_mrr": new_mrr,
                "delta_mrr": new_mrr - old_mrr,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info("  Queried %d/%d", i + 1, n)

    mgr.close()

    # ================================================================
    # ANALYSIS
    # ================================================================

    # Overall stats
    old_overall = statistics.mean([r["old_mrr"] for r in rows])
    new_overall = statistics.mean([r["new_mrr"] for r in rows])
    old_exac = statistics.mean(
        [r["old_mrr"] for r in rows if r["qtype"] == "exact_token"]
    )
    new_exac = statistics.mean(
        [r["new_mrr"] for r in rows if r["qtype"] == "exact_token"]
    )
    old_conc = statistics.mean(
        [r["old_mrr"] for r in rows if r["qtype"] == "conceptual"]
    )
    new_conc = statistics.mean(
        [r["new_mrr"] for r in rows if r["qtype"] == "conceptual"]
    )

    print()
    print("=" * 70)
    print("  OLD vs NEW RRF: PER-QUERY COMPARISON")
    print("=" * 70)
    print(
        f"  Old RRF: w_d={OLD_WD}, w_s={OLD_WS}   New RRF: w_d={NEW_WD}, w_s={NEW_WS}"
    )
    print()
    print(f"  {'':>20}  {'Old MRR':>8}  {'New MRR':>8}  {'Δ':>8}")
    print(f"  {'-' * 20}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    print(
        f"  {'Overall':>20}  {old_overall:>8.4f}  {new_overall:>8.4f}  {new_overall - old_overall:>+8.4f}"
    )
    print(
        f"  {'Exact_token':>20}  {old_exac:>8.4f}  {new_exac:>8.4f}  {new_exac - old_exac:>+8.4f}"
    )
    print(
        f"  {'Conceptual':>20}  {old_conc:>8.4f}  {new_conc:>8.4f}  {new_conc - old_conc:>+8.4f}"
    )

    # Queries improved, unchanged, regressed
    improved = [r for r in rows if r["delta_mrr"] > 0]
    unchanged = [r for r in rows if r["delta_mrr"] == 0]
    regressed = [r for r in rows if r["delta_mrr"] < 0]

    print(f"\n  {'Improved':>10}: {len(improved)} queries")
    print(f"  {'Unchanged':>10}: {len(unchanged)} queries")
    print(f"  {'Regressed':>10}: {len(regressed)} queries")

    # ================================================================
    # SPARSE-SAVED QUERIES: where old RRF succeeded because of sparse
    # ================================================================
    print()
    print("=" * 70)
    print("  SPARSE-SAVED QUERIES (dense missed, sparse found, old RRF worked)")
    print("=" * 70)

    sparse_saved = [
        r
        for r in rows
        if not r["d_found"] and r["s_found"] and r["old_rank"] is not None
    ]
    for r in sparse_saved:
        regressed_flag = (
            " <<< REGRESSED" if r["delta_mrr"] < 0 else " (still same or better)"
        )
        print(
            f"  Q{r['qnum']:>2} [{r['qtype']}] dense=miss, sparse=#{r['s_rank']}, old=#{r['old_rank']}, new=#{r['new_rank']}{regressed_flag}"
        )
        print(f"       {r['query'][:90]}")

    if not sparse_saved:
        print("  (none — no sparse-saved queries in this set)")

    # Also: queries where dense found and sparse didn't, and old RRF worked
    dense_saved = [
        r
        for r in rows
        if r["d_found"] and not r["s_found"] and r["old_rank"] is not None
    ]

    # ================================================================
    # REGRESSIONS: queries that got worse
    # ================================================================
    print()
    print("=" * 70)
    print("  REGRESSIONS (new RRF worse than old)")
    print("=" * 70)

    if regressed:
        regressed.sort(key=lambda r: r["delta_mrr"])
        for r in regressed:
            d_status = f"#{r['d_rank']}" if r["d_found"] else "miss"
            s_status = f"#{r['s_rank']}" if r["s_found"] else "miss"
            print(
                f"  Q{r['qnum']:>2} [{r['qtype']}] dense={d_status}, sparse={s_status}  old=#{r['old_rank']}→new=#{r['new_rank']}  Δ={r['delta_mrr']:+.4f}"
            )
            print(f"       {r['query'][:90]}")
    else:
        print("  (none — no regressions)")

    # ================================================================
    # IMPROVEMENTS: queries that got better
    # ================================================================
    print()
    print("=" * 70)
    print("  ALL IMPROVEMENTS (new RRF better than old)")
    print("=" * 70)

    if improved:
        improved.sort(key=lambda r: r["delta_mrr"], reverse=True)
        for r in improved:
            d_status = f"#{r['d_rank']}" if r["d_found"] else "miss"
            s_status = f"#{r['s_rank']}" if r["s_found"] else "miss"
            print(
                f"  Q{r['qnum']:>2} [{r['qtype']}] dense={d_status}, sparse={s_status}  old=#{r['old_rank']}→new=#{r['new_rank']}  Δ={r['delta_mrr']:+.4f}"
            )
            print(f"       {r['query'][:90]}")
    else:
        print("  (none)")

    # ================================================================
    # CASES WHERE BOTH DENSE AND SPARSE MISSED (first-stage recall failure)
    # ================================================================
    print()
    print("=" * 70)
    print("  FIRST-STAGE RECALL FAILURES (both dense and sparse missed)")
    print("=" * 70)
    both_missed = [r for r in rows if not r["d_found"] and not r["s_found"]]
    if both_missed:
        for r in both_missed:
            print(
                f"  Q{r['qnum']:>2} [{r['qtype']}] old=#{r['old_rank']}, new=#{r['new_rank']}"
            )
            print(f"       {r['query'][:90]}")
    else:
        print("  (none — all queries find gold in at least one retriever)")


if __name__ == "__main__":
    main()
