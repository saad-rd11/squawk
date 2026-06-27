"""
Evaluate retrieval: parent-collapsed MRR, multi-ACN completeness, per-query breakdown.

Aggregates child chunks to parent ACN (max score), then measures MRR over
parents. This removes chunk-count contamination and matches the retrieval you
will ship (one incident per result row).

Usage:
    python -m pipeline.eval_recall
"""

import json
import logging
import re

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KS = [1, 5, 10, 20, 50]
SEARCH_LIMIT = 200
RRF_K = 60
# Dense-weighted RRF: ratio 3:2 dense:sparse (conservative — preserves all sparse saves)
W_DENSE = 0.60
W_SPARSE = 0.40


def _collapse_to_parents(scored_points) -> list[tuple[int, float]]:
    """Group child points by ACN, keep max score per parent.
    Returns list of (acn, max_score) sorted by score descending."""
    acn_scores: dict[int, float] = {}
    for sp in scored_points:
        rid = sp.payload.get("report_id")
        if rid is not None:
            acn = int(rid)
            score = sp.score
            if acn not in acn_scores or score > acn_scores[acn]:
                acn_scores[acn] = score
    return sorted(acn_scores.items(), key=lambda x: x[1], reverse=True)


def _rrf_combined(
    dense: list[tuple[int, float]], sparse: list[tuple[int, float]]
) -> list[tuple[int, float]]:
    """Dense-weighted reciprocal rank fusion over parent-level rankings.

    Using W_DENSE=0.80, W_SPARSE=0.20 (4:1 ratio) from global weight sweep.
    Fixes the heart-attack→battery fusion artifact where sparse confidently
    ranks a wrong result and overrides correct dense.
    """

    def _build_ranks(collapsed):
        return {acn: rank for rank, (acn, _) in enumerate(collapsed, 1)}

    dr = _build_ranks(dense)
    sr = _build_ranks(sparse)
    all_acns = set(dr) | set(sr)
    default = max(len(dense), len(sparse)) + 1

    scores = {}
    for acn in all_acns:
        scores[acn] = W_DENSE / (RRF_K + dr.get(acn, default)) + W_SPARSE / (
            RRF_K + sr.get(acn, default)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _parent_mrr(collapsed: list[tuple[int, float]], expected: set[int]) -> float:
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn in expected:
            return 1.0 / rank
    return 0.0


def _parent_completeness(
    collapsed: list[tuple[int, float]], expected: set[int], k: int
) -> float:
    found = set(acn for acn, _ in collapsed[:k]) & expected
    return len(found) / len(expected) if expected else 0.0


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    has_types = all("type" in q for q in queries)
    n = len(queries)
    logger.info("Loaded %d queries (%s)", n, "typed" if has_types else "untyped")

    dense_embedder = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sparse_embedder = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )

    query_texts = [q["query"] for q in queries]
    logger.info("Embedding queries (dense)...")
    dense_vecs = dense_embedder.embed(query_texts)
    logger.info("Embedding queries (sparse)...")
    sparse_vecs = sparse_embedder.embed(query_texts)

    mgr = Stage3Collection(path="./qdrant_storage")
    info = mgr.client.get_collection(COLLECTION_NAME)
    logger.info("Collection: %d points", info.points_count)

    # Accumulators
    dense_rr: list[float] = []
    sparse_rr: list[float] = []
    combined_rr: list[float] = []

    multi_qs = 0
    multi_dense_comp = {k: 0.0 for k in KS}
    multi_sparse_comp = {k: 0.0 for k in KS}
    multi_combined_comp = {k: 0.0 for k in KS}

    type_multi: dict[str, dict] = {}
    type_multi_qs: dict[str, int] = {}
    for q in queries:
        t = q.get("type", "conceptual")
        if t not in type_multi:
            type_multi[t] = {
                "dense": {k: 0.0 for k in KS},
                "sparse": {k: 0.0 for k in KS},
                "combined": {k: 0.0 for k in KS},
            }
            type_multi_qs[t] = 0

    detail_rows: list[list] = []

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        dvec = dense_vecs[i]
        svec = sparse_vecs[i]
        qtype = q.get("type", "conceptual")

        # Dense
        dense_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dvec,
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        dense_collapsed = _collapse_to_parents(dense_raw)

        # Sparse
        sparse_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(indices=list(svec.keys()), values=list(svec.values())),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_collapsed = _collapse_to_parents(sparse_raw)

        # Combined (RRF)
        combined_collapsed = _rrf_combined(dense_collapsed, sparse_collapsed)

        # MRR
        drr = _parent_mrr(dense_collapsed, expected)
        srr = _parent_mrr(sparse_collapsed, expected)
        crr = _parent_mrr(combined_collapsed, expected)

        dense_rr.append(drr)
        sparse_rr.append(srr)
        combined_rr.append(crr)

        # Multi-ACN completeness
        is_multi = len(expected) > 1
        if is_multi:
            multi_qs += 1
            for k in KS:
                multi_dense_comp[k] += _parent_completeness(
                    dense_collapsed, expected, k
                )
                multi_sparse_comp[k] += _parent_completeness(
                    sparse_collapsed, expected, k
                )
                multi_combined_comp[k] += _parent_completeness(
                    combined_collapsed, expected, k
                )

            t = q.get("type", "conceptual")
            type_multi_qs[t] += 1
            for k in KS:
                type_multi[t]["dense"][k] += _parent_completeness(
                    dense_collapsed, expected, k
                )
                type_multi[t]["sparse"][k] += _parent_completeness(
                    sparse_collapsed, expected, k
                )
                type_multi[t]["combined"][k] += _parent_completeness(
                    combined_collapsed, expected, k
                )

        # Per-query row
        dense_rank = int(1 / drr) if drr > 0 else None
        sparse_rank = int(1 / srr) if srr > 0 else None
        detail_rows.append(
            [
                i + 1,
                qtype[:4],
                len(expected),
                dense_rank,
                sparse_rank,
                f"{drr:.3f}",
                f"{srr:.3f}",
                q["query"][:80],
            ]
        )

        if (i + 1) % 5 == 0 or i == n - 1:
            logger.info("Processed %d/%d", i + 1, n)

    # Build MULTI_BY_TYPE list
    MULTI_BY_TYPE = []
    for t in sorted(type_multi):
        MULTI_BY_TYPE.append(
            (
                t,
                type_multi_qs[t],
                type_multi[t]["dense"],
                type_multi[t]["sparse"],
                type_multi[t]["combined"],
            )
        )

    # Leakage benchmark constants
    LEAKAGE_PCT = 35.3
    TOP_LEAKERS = [
        (
            "Non-aircraft tarmac traffic ignoring right-of-way rules at intersections, compelling the flight deck to perform harsh stops to prevent a ground collision.",
            56.5,
        ),
        (
            "A teacher resorting to hitting the learner in the cockpit to regain authority over the control column just as the aircraft was trying to unstick from the tarmac.",
            45.5,
        ),
        (
            "The pilot monitoring failing to catch that the flight guidance computer bypassed a vertical waypoint constraint because they were utilizing a different thrust/pitch logic to slow down.",
            44.0,
        ),
    ]
    TOP_LEAKER_TEXTS = {t[0] for t in TOP_LEAKERS}

    # ================================================================
    #  OUTPUT
    # ================================================================
    print()
    print("=" * 70)
    print(
        f"  PARENT-COLLAPSED EVAL  —  {n} queries  /  {info.points_count} pool points  /  {RRF_K=}"
    )
    print("=" * 70)

    # Leakage caveat
    print()
    print("  LEAKAGE BENCHMARK (conceptual queries only)")
    print(f"    Avg query-narrative token overlap: {LEAKAGE_PCT:.1f}%")
    print("    Top 3 individual overlaps (where sparse advantage is most discounted):")
    for q_text, pct in TOP_LEAKERS:
        print(f"      {pct:.1f}%  {q_text[:65]}")
    print("    (Note: 6 aircraft-code queries removed per earlier gate —")
    print("     slated for Stage 5 structured-filter tests)")

    # MRR overall
    print()
    print("  MRR (Mean Reciprocal Rank) — parent-collapsed")
    print(f"    Overall  ({n} queries)")
    print(f"      Dense:    {sum(dense_rr) / n:.4f}")
    print(f"      Sparse:   {sum(sparse_rr) / n:.4f}")
    print(f"      Combined: {sum(combined_rr) / n:.4f}  (RRF, k={RRF_K})")

    # MRR by type
    if has_types:
        for qtype in sorted(set(q.get("type", "conceptual") for q in queries)):
            idxs = [
                i for i, q in enumerate(queries) if q.get("type", "conceptual") == qtype
            ]
            mrr_d = sum(dense_rr[i] for i in idxs) / len(idxs)
            mrr_s = sum(sparse_rr[i] for i in idxs) / len(idxs)
            mrr_c = sum(combined_rr[i] for i in idxs) / len(idxs)
            print(f"    [{qtype}] ({len(idxs)} queries)")
            print(f"      Dense:    {mrr_d:.4f}")
            print(f"      Sparse:   {mrr_s:.4f}")
            print(f"      Combined: {mrr_c:.4f}")

    # Multi-ACN completeness overall + by type
    for title, qs_count, dc, sc, cc in [
        ("overall", multi_qs, multi_dense_comp, multi_sparse_comp, multi_combined_comp)
    ] + MULTI_BY_TYPE:
        if qs_count == 0:
            continue
        print()
        print(
            f"  Multi-ACN completeness recall — {title} ({qs_count} queries with 2+ expected ACNs)"
        )
        print(f"    {'k':>5}  {'Dense':>8}  {'Sparse':>8}  {'Combined':>8}")
        print("    " + "-" * 32)
        for k in KS:
            print(
                f"    {k:>5}  {dc[k] / qs_count * 100:>7.1f}%  {sc[k] / qs_count * 100:>7.1f}%  {cc[k] / qs_count * 100:>7.1f}%"
            )

    # Per-query table
    print()
    print("  Per-query detail (parent-collapsed ranks)")
    print(
        f"    {'Q':>3} {'Type':>4} {'#Exp':>4}  {'Dense@1':>7} {'Sparse@1':>8}  {'MRR-d':>6} {'MRR-s':>6}  Query"
    )
    print("    " + "-" * 100)
    for row in detail_rows:
        qnum, qtype, nexp, dr, sr, mrr_d, mrr_s, qtext = row
        dr_str = f"#{dr}" if dr else "miss"
        sr_str = f"#{sr}" if sr else "miss"
        leak_flag = " << leak" if qtext.rstrip() in TOP_LEAKER_TEXTS else ""
        print(
            f"    {qnum:>3} {qtype:>4} {nexp:>4}  {dr_str:>7} {sr_str:>8}  {mrr_d:>6} {mrr_s:>6}  {qtext}{leak_flag}"
        )

    # Restate leakage
    print()
    print(f"  Caveat: conceptual leakage = {LEAKAGE_PCT:.1f}% token overlap")
    print(
        "  (reduced from 56% in the original set; residual is topical vocabulary, not lifted phrasing)"
    )

    mgr.close()


if __name__ == "__main__":
    main()
