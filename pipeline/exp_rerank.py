"""
Experiment: cross-encoder reranker over dense top-K parents.

Loads a cross-encoder model, retrieves dense top-N parents from Qdrant,
reranks their best child chunk with the cross-encoder, and measures MRR.

Usage:
    python -m pipeline.exp_rerank
"""

import json
import logging
import sys
from time import time

from sentence_transformers import CrossEncoder

from qdrant_client.models import SparseVector
from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KS = [1, 5, 10, 20, 50]
SEARCH_LIMIT = 200
RERANK_DEPTH = 50
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _collapse_to_parents(scored_points) -> list[tuple[int, float, str]]:
    """Group child points by ACN, keep max score + the best chunk text per parent."""
    acn_scores: dict[int, tuple[float, str]] = {}
    for sp in scored_points:
        rid = sp.payload.get("report_id")
        if rid is not None:
            acn = int(rid)
            score = sp.score
            chunk = sp.payload.get("chunk", "")
            prefix = sp.payload.get("context_prefix", "")
            text = f"[{prefix}]\nNarrative: {chunk}" if prefix else chunk
            if acn not in acn_scores or score > acn_scores[acn][0]:
                acn_scores[acn] = (score, text)
    # Sort by dense score descending
    return [
        (acn, score, text)
        for acn, (score, text) in sorted(
            acn_scores.items(), key=lambda x: x[1][0], reverse=True
        )
    ]


def _mrr(collapsed, expected):
    for rank, (acn, _, _) in enumerate(collapsed, 1):
        if acn in expected:
            return 1.0 / rank
    return 0.0


def _completeness(collapsed, expected, k):
    found = set(acn for acn, _, _ in collapsed[:k]) & expected
    return len(found) / len(expected) if expected else 0.0


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    n = len(queries)
    logger.info("Loaded %d queries", n)

    # Load reranker
    logger.info("Loading reranker: %s ...", RERANK_MODEL)
    t0 = time()
    reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
    logger.info("Reranker loaded in %.1fs", time() - t0)

    # Embedders for query embeddings (needed for dense search)
    dense_embedder = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    query_texts = [q["query"] for q in queries]
    dense_vecs = dense_embedder.embed(query_texts)

    mgr = Stage3Collection(path="./qdrant_storage")
    info = mgr.client.get_collection(COLLECTION_NAME)
    logger.info("Collection: %d points", info.points_count)

    baseline_rr: list[float] = []
    rerank_rr: list[float] = []
    detail_rows: list[list] = []

    baseline_multi = {k: 0.0 for k in KS}
    rerank_multi = {k: 0.0 for k in KS}
    multi_qs = 0

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        dvec = dense_vecs[i]
        qtext = q["query"]
        qtype = q.get("type", "conceptual")

        # Dense search
        dense_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dvec,
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id", "chunk", "context_prefix"],
        ).points
        dense_parents = _collapse_to_parents(dense_raw)

        # Baseline MRR (dense parent-collapsed)
        baseline_mrr_val = _mrr(dense_parents, expected)
        baseline_rr.append(baseline_mrr_val)

        # Rerank: take top RERANK_DEPTH parents, score with cross-encoder
        candidates = dense_parents[:RERANK_DEPTH]
        if candidates:
            pairs = [(qtext, text) for _, _, text in candidates]
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            # Re-sort by CE score descending; keep original dense order for ties
            reranked = sorted(
                zip(candidates, ce_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            reranked_parents = [cand for cand, _ in reranked]
        else:
            reranked_parents = candidates

        rerank_mrr_val = _mrr(reranked_parents, expected)
        rerank_rr.append(rerank_mrr_val)

        # Multi-ACN completeness
        if len(expected) > 1:
            multi_qs += 1
            for k in KS:
                baseline_multi[k] += _completeness(dense_parents, expected, k)
                rerank_multi[k] += _completeness(reranked_parents, expected, k)

        # Detail row
        dense_rank = int(1 / baseline_mrr_val) if baseline_mrr_val > 0 else None
        rerank_rank = int(1 / rerank_mrr_val) if rerank_mrr_val > 0 else None
        detail_rows.append(
            [
                i + 1,
                qtype[:4],
                len(expected),
                dense_rank,
                rerank_rank,
                f"{baseline_mrr_val:.4f}",
                f"{rerank_mrr_val:.4f}",
                qtext[:80],
            ]
        )

        if (i + 1) % 5 == 0 or i == n - 1:
            logger.info("Processed %d/%d", i + 1, n)

    # Per-type accumulators
    type_base: dict[str, list[float]] = {}
    type_rerank: dict[str, list[float]] = {}
    for i, q in enumerate(queries):
        t = q.get("type", "conceptual")
        type_base.setdefault(t, []).append(baseline_rr[i])
        type_rerank.setdefault(t, []).append(rerank_rr[i])

    # Print results
    print()
    print("=" * 70)
    print(f"  CROSS-ENCODER RERANK EXPERIMENT")
    print(f"  Model: {RERANK_MODEL}")
    print(f"  Rerank depth: top-{RERANK_DEPTH} parents per query")
    print(f"  {n} queries  /  {info.points_count} pool points")
    print("=" * 70)

    # MRR overall
    print()
    print("  MRR (parent-collapsed)")
    print(f"    Baseline (dense only):      {sum(baseline_rr) / n:.4f}")
    print(f"    + CE reranker top-{RERANK_DEPTH}:  {sum(rerank_rr) / n:.4f}")
    print(
        f"    Δ:                           {sum(rerank_rr) / n - sum(baseline_rr) / n:+.4f}"
    )

    # MRR by type
    print()
    for t in sorted(type_base):
        b = type_base[t]
        r = type_rerank[t]
        print(f"    [{t}] ({len(b)} queries)")
        print(f"      Baseline:  {sum(b) / len(b):.4f}")
        print(f"      + Rerank:   {sum(r) / len(r):.4f}")
        print(f"      Δ:          {sum(r) / len(r) - sum(b) / len(b):+.4f}")

    # Multi-ACN completeness
    if multi_qs > 0:
        print()
        print(
            f"  Multi-ACN completeness recall ({multi_qs} queries with 2+ expected ACNs)"
        )
        print(f"    {'k':>5}  {'Dense':>8}  {'+Rerank':>8}")
        print("    " + "-" * 24)
        for k in KS:
            db = baseline_multi[k] / multi_qs * 100
            dr = rerank_multi[k] / multi_qs * 100
            print(f"    {k:>5}  {db:>7.1f}%  {dr:>7.1f}%")

    # Per-query detail
    print()
    print("  Per-query detail")
    print(
        f"    {'Q':>3} {'Type':>4} {'#Exp':>4}  {'Base@1':>6} {'Rer@1':>6}  {'MRR-b':>6} {'MRR-r':>6}  Query"
    )
    print("    " + "-" * 100)
    for row in detail_rows:
        qnum, qtype, nexp, dr, rr, mrr_b, mrr_r, qtext = row
        dr_str = f"#{dr}" if dr else "miss"
        rr_str = f"#{rr}" if rr else "miss"
        gain = "+" if rr and dr and rr < dr else ("=" if rr == dr else "")
        print(
            f"    {qnum:>3} {qtype:>4} {nexp:>4}  {dr_str:>6} {rr_str:>6}  {mrr_b:>6} {mrr_r:>6}  {qtext} {gain}"
        )

    mgr.close()


if __name__ == "__main__":
    main()
