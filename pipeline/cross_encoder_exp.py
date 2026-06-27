"""
Cross-encoder reranker experiment.
Two-stage: bi-encoder + sparse → RRF → top-20 → cross-encoder rerank.
"""

import json
import logging
import time
import numpy as np

from qdrant_client.models import SparseVector
from sentence_transformers import CrossEncoder

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RRF_K = 60
SEARCH_LIMIT = 200
RERANK_DEPTH = 30  # how many top RRF results to rerank


def collapse(raw):
    out = {}
    for sp in raw:
        rid = sp.payload.get("report_id")
        if rid is not None:
            a = int(rid)
            if a not in out or sp.score > out[a]["score"]:
                out[a] = {"score": sp.score}
    return sorted(out.items(), key=lambda x: x[1]["score"], reverse=True)


def main():
    with open("eval_queries.json") as f:
        queries = json.load(f)

    import pandas as pd

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)
    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))
    pool_df = df[df["ACN"].isin(pool_acns)].set_index("ACN")

    config = DEFAULT_CONFIG

    # Stage 1 encoders
    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )
    mgr = Stage3Collection(path="./qdrant_storage")

    qtexts = [q["query"] for q in queries]
    logger.info("Embedding queries (dense + sparse)...")
    dvecs = ded.embed(qtexts)
    svecs = sed.embed(qtexts)

    # Stage 2: cross-encoder (downloads model on first run)
    logger.info("Loading cross-encoder model...")
    t0 = time.time()
    cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info(f"Cross-encoder loaded in {time.time() - t0:.1f}s")

    # Accumulators
    baseline_rr = []
    reranked_rr = []
    per_query = []

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        qnum = i + 1
        qtext = q["query"]
        qtype = q.get("type", "conceptual")

        # Stage 1: retrieve from Qdrant
        dense_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dvecs[i],
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(
                indices=list(svecs[i].keys()), values=list(svecs[i].values())
            ),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points

        dense_collapsed = collapse(dense_raw)
        sparse_collapsed = collapse(sparse_raw)

        dr_map = {a: r for r, (a, _) in enumerate(dense_collapsed, 1)}
        sr_map = {a: r for r, (a, _) in enumerate(sparse_collapsed, 1)}
        default = max(len(dense_collapsed), len(sparse_collapsed)) + 1
        all_a = set(dr_map) | set(sr_map)
        scores = {}
        for a in all_a:
            scores[a] = 1.0 / (RRF_K + dr_map.get(a, default)) + 1.0 / (
                RRF_K + sr_map.get(a, default)
            )
        rrf_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Baseline MRR
        baseline_mrr = 0.0
        for rank, (acn, _) in enumerate(rrf_results, 1):
            if acn in expected:
                baseline_mrr = 1.0 / rank
                break
        baseline_rr.append(baseline_mrr)

        # Cross-encoder reranking
        top_k = rrf_results[:RERANK_DEPTH]
        pairs = []
        candidate_acns = []
        for acn, _ in top_k:
            try:
                syn = str(pool_df.loc[acn, "Synopsis"])
            except KeyError:
                syn = ""
            pairs.append((qtext, syn))
            candidate_acns.append(acn)

        if pairs:
            ce_scores = cross_enc.predict(pairs, show_progress_bar=False)
            reranked = sorted(
                zip(candidate_acns, ce_scores), key=lambda x: x[1], reverse=True
            )
        else:
            reranked = top_k

        # Reranked MRR
        reranked_mrr = 0.0
        for rank, (acn, _) in enumerate(reranked, 1):
            if acn in expected:
                reranked_mrr = 1.0 / rank
                break
        reranked_rr.append(reranked_mrr)

        per_query.append(
            {
                "qnum": qnum,
                "query": qtext[:60],
                "qtype": qtype,
                "expected": sorted(expected),
                "baseline_mrr": baseline_mrr,
                "reranked_mrr": reranked_mrr,
                "improved": reranked_mrr > baseline_mrr,
                "degraded": reranked_mrr < baseline_mrr,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(queries)}")

    mgr.close()

    n = len(queries)
    overall_baseline = sum(baseline_rr) / n
    overall_reranked = sum(reranked_rr) / n

    # ── Report ──
    print()
    print("=" * 75)
    print("  CROSS-ENCODER RERANKER EXPERIMENT")
    print("=" * 75)
    print()
    print(f"  Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"  Rerank depth: top-{RERANK_DEPTH} from RRF")
    print(f"  Queries: {n}")
    print()
    print(f"  {'':>30}  {'Baseline':>10}  {'Reranked':>10}  {'Δ':>8}")
    print(f"  {'-' * 60}")
    print(
        f"  {'Overall MRR':>30}  {overall_baseline:.4f}  {overall_reranked:.4f}  {overall_reranked - overall_baseline:+.4f}"
    )

    # By type
    for qtype in sorted(set(q["qtype"] for q in per_query)):
        subset = [q for q in per_query if q["qtype"] == qtype]
        b = sum(q["baseline_mrr"] for q in subset) / len(subset)
        r = sum(q["reranked_mrr"] for q in subset) / len(subset)
        print(
            f"  {f'[{qtype}] ({len(subset)} queries)':>30}  {b:.4f}  {r:.4f}  {r - b:+.4f}"
        )

    # Counts
    n_improved = sum(1 for q in per_query if q["improved"])
    n_degraded = sum(1 for q in per_query if q["degraded"])
    n_same = n - n_improved - n_degraded
    print(
        f"  {'Improved/Degraded/Same':>30}  {n_improved:>4}/{n_degraded:>4}/{n_same:>4}"
    )

    # ── Per-query detail (show interesting cases) ──
    print()
    print("  ── Per-query detail (all queries) ──")
    print(f"  {'Q':>3} {'Type':>6}  {'Base MRR':>8} {'Rerank MRR':>10} {'Δ':>8}  Query")
    print(f"  {'-' * 75}")
    for q in per_query:
        delta = q["reranked_mrr"] - q["baseline_mrr"]
        marker = ""
        if delta > 0:
            marker = " ↑"
        elif delta < 0:
            marker = " ↓"
        print(
            f"  {q['qnum']:>3} {q['qtype'][:6]:>6}  {q['baseline_mrr']:.4f}  {q['reranked_mrr']:.4f}  {delta:+.4f}{marker}  {q['query']}"
        )

    # ── Spotlight: Q2 (permission vs clearance) ──
    print()
    print(
        "  ── Spotlight: Q4 = 'pilots who landed without permission' (evaluated Q2) ──"
    )
    q2 = [q for q in per_query if q["qnum"] == 2]
    if q2:
        q2 = q2[0]
        print(f"  Baseline MRR: {q2['baseline_mrr']:.4f}")
        print(f"  Reranked MRR: {q2['reranked_mrr']:.4f}")
        print(f"  Delta:        {q2['reranked_mrr'] - q2['baseline_mrr']:+.4f}")

    # ── Summary ──
    print()
    print("=" * 75)
    print(
        f"  Overall: {overall_baseline:.4f} → {overall_reranked:.4f} (Δ={overall_reranked - overall_baseline:+.4f})"
    )
    print(f"  {n_improved} queries improved, {n_degraded} degraded, {n_same} unchanged")
    print("=" * 75)


if __name__ == "__main__":
    main()
