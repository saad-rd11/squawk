"""
Reranker test: use answerdotai/answerai-colbert-small-v1 to rerank top-50
candidates for the 15 low-leakage conceptual queries.
Compares combined MRR before vs after ColBERT late-interaction reranking.

Run:  ./venv/bin/python -m pipeline.reranker_test
"""

import json
import logging
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import Stage3Collection, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 300
RRF_K = 60
RERANK_CUTOFF = 50
LEAKAGE_THRESHOLD = 0.353


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


def compute_leakage(query_text, narrative_text):
    qt = set(query_text.lower().split())
    nt = set(narrative_text.lower().split())
    if not qt:
        return 0.0
    return len(qt & nt) / len(qt)


def maxsim_score(q_emb, p_emb):
    """ColBERT late interaction: sum over query tokens of max over doc tokens."""
    q_arr = np.array(q_emb)  # [q_tokens, dim]
    p_arr = np.array(p_emb)  # [p_tokens, dim]
    sim = q_arr @ p_arr.T  # [q_tokens, p_tokens]
    return float(sim.max(axis=1).sum())


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)
    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))

    # Load narratives for reranking
    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)
    pool_df = df[df["ACN"].isin(pool_acns)].set_index("ACN")

    # Embedders for candidate retrieval
    dense_embedder = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sparse_embedder = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )

    # ColBERT reranker
    from fastembed.late_interaction import LateInteractionTextEmbedding

    colbert = LateInteractionTextEmbedding(
        model_name="answerdotai/answerai-colbert-small-v1"
    )
    logger.info("ColBERT dim=%d", colbert.embedding_size)

    mgr = Stage3Collection(path="./qdrant_storage")

    query_texts = [q["query"] for q in queries]
    logger.info("Embedding queries for candidate retrieval...")
    dense_vecs = dense_embedder.embed(query_texts)
    sparse_vecs = sparse_embedder.embed(query_texts)
    logger.info("Done retrieval embeddings")

    # ─── Per-query: candidate retrieval + reranking ───
    before_results = []  # combined MRR before reranking
    after_results = []  # ColBERT reranked MRR
    low_leak_details = []  # per-query detail

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        qtype = q.get("type", "conceptual")
        gold_acn = next(iter(expected), None)

        # Decide if low-leakage conceptual
        if qtype != "conceptual":
            continue  # only test conceptual queries
        narrs = []
        for acn in expected:
            try:
                narrs.append(str(pool_df.loc[acn, "Narrative"]))
            except KeyError:
                pass
        full_narr = " ".join(narrs)
        leakage = compute_leakage(q["query"], full_narr)

        if leakage > LEAKAGE_THRESHOLD:
            continue  # only low-leakage

        # ── Candidate retrieval ──
        dvec = dense_vecs[i]
        svec = sparse_vecs[i]

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

        dense_ranks = {acn: rank for rank, (acn, _) in enumerate(dense_collapsed, 1)}
        sparse_ranks = {acn: rank for rank, (acn, _) in enumerate(sparse_collapsed, 1)}
        all_acns = set(dense_ranks) | set(sparse_ranks)
        default = max(len(dense_collapsed), len(sparse_collapsed)) + 1

        # RRF scores
        scores = {}
        for acn in all_acns:
            scores[acn] = 1.0 / (RRF_K + dense_ranks.get(acn, default)) + 1.0 / (
                RRF_K + sparse_ranks.get(acn, default)
            )
        combined_collapsed = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # MRR before reranking (top-50)
        combined_before = combined_collapsed[:RERANK_CUTOFF]
        before_mrr = _parent_mrr(combined_before, expected)

        # ── ColBERT reranking ──
        # Get parent texts for the top-50 candidates
        candidate_acns = [acn for acn, _ in combined_before]
        candidate_texts = []
        for acn in candidate_acns:
            try:
                candidate_texts.append(str(pool_df.loc[acn, "Narrative"]))
            except KeyError:
                candidate_texts.append("")

        # Embed query with ColBERT
        q_emb = list(colbert.query_embed([q["query"]]))[0]

        # Embed candidate passages in batches
        colbert_scores = {}
        batch_size = 16
        for start in range(0, len(candidate_texts), batch_size):
            batch = candidate_texts[start : start + batch_size]
            batch_acns = candidate_acns[start : start + batch_size]
            p_embs = list(colbert.passage_embed(batch))
            for acn, p_emb in zip(batch_acns, p_embs):
                colbert_scores[acn] = maxsim_score(q_emb, p_emb)

        # Re-rank by ColBERT score
        reranked = sorted(colbert_scores.items(), key=lambda x: x[1], reverse=True)
        after_mrr = _parent_mrr(reranked, expected)

        # ── Tracking ──
        gold_rank_before = None
        for rank, (acn, _) in enumerate(combined_before, 1):
            if acn == gold_acn:
                gold_rank_before = rank
                break
        gold_rank_after = None
        for rank, (acn, _) in enumerate(reranked, 1):
            if acn == gold_acn:
                gold_rank_after = rank
                break

        before_results.append(before_mrr)
        after_results.append(after_mrr)
        low_leak_details.append(
            {
                "qnum": i + 1,
                "query": q["query"][:60],
                "leakage": leakage,
                "gold_rank_before": gold_rank_before,
                "gold_rank_after": gold_rank_after,
                "mrr_before": before_mrr,
                "mrr_after": after_mrr,
            }
        )

        if (len(low_leak_details)) % 5 == 0:
            logger.info("Reranked %d/%d low-leak queries", len(low_leak_details), 15)

    mgr.close()

    # ─── Results ───
    n = len(low_leak_details)
    print(f"\n{'=' * 80}")
    print(f"COLBERT RERANKER TEST — Low-Leakage Conceptual Queries (n={n})")
    print(f"{'=' * 80}")

    print(
        f"\n{'Q#':>4} {'Leak%':>6} {'C@1':>5} {'Col@1':>6} {'MRRbef':>7} {'MRRaft':>7} {'ΔMRR':>6} {'Improve?':>8}  Query"
    )
    print("-" * 110)
    improved = 0
    worsened = 0
    for r in sorted(low_leak_details, key=lambda x: x["qnum"]):
        gb = r["gold_rank_before"] if r["gold_rank_before"] else "miss"
        ga = r["gold_rank_after"] if r["gold_rank_after"] else "miss"
        delta = r["mrr_after"] - r["mrr_before"]
        imp = "YES" if delta > 0 else ("NO" if delta < 0 else "tie")
        if delta > 0:
            improved += 1
        elif delta < 0:
            worsened += 1
        print(
            f"{r['qnum']:>4} {100 * r['leakage']:>5.1f}% {str(gb):>5} {str(ga):>6} "
            f"{r['mrr_before']:.4f} {r['mrr_after']:.4f} {delta:+.4f} {imp:>8}  {r['query']}"
        )

    avg_before = statistics.mean(before_results)
    avg_after = statistics.mean(after_results)
    print(f"\n--- Summary (low-leakage conceptual, n={n}) ---")
    print(f"  Combined MRR before reranking: {avg_before:.4f}")
    print(f"  ColBERT reranked MRR:          {avg_after:.4f}")
    print(f"  ΔMRR:                          {avg_after - avg_before:+.4f}")
    print(
        f"  Improved: {improved}/{n}  Worsened: {worsened}/{n}  Tied: {n - improved - worsened}/{n}"
    )

    # Also compute by rerankability
    reachable = [r for r in low_leak_details if r["gold_rank_before"] is not None]
    n_reachable = len(reachable)
    if n_reachable:
        reachable_before = statistics.mean([r["mrr_before"] for r in reachable])
        reachable_after = statistics.mean([r["mrr_after"] for r in reachable])
        unreachable = [r for r in low_leak_details if r["gold_rank_before"] is None]
        print(
            f"\n  Gold in top-50: {n_reachable}/{n} "
            f"(MRR before={reachable_before:.4f}, after={reachable_after:.4f}, "
            f"Δ={reachable_after - reachable_before:+.4f})"
        )
        if unreachable:
            print(
                f"  Gold NOT in top-50: {len(unreachable)} queries (reranker cannot help; MRR=0)"
            )

    # Leakage stratified by tier
    print(f"\n--- Leakage Stratification (all conceptual, k=60 combined) ---")
    all_conc_before = []
    all_conc_after = []
    high_leak_before = []
    high_leak_after = []
    low_leak_before = []
    low_leak_after = []

    for i, q in enumerate(queries):
        if q.get("type") != "conceptual":
            continue
        narrs = []
        for acn in q["expected_acns"]:
            try:
                narrs.append(str(pool_df.loc[acn, "Narrative"]))
            except KeyError:
                pass
        full_narr = " ".join(narrs)
        leakage = compute_leakage(q["query"], full_narr)
        if leakage <= LEAKAGE_THRESHOLD:
            idx = len(low_leak_before)
            if idx < len(before_results):
                low_leak_before.append(before_results[idx])
                low_leak_after.append(after_results[idx])
        else:
            # For high-leak, we didn't rerank — use combined MRR from before
            high_leak_before.append(0)
            high_leak_after.append(0)

    # Actually, let me just report what we have from the diagnosis run
    # for high-leakage, since we didn't rerank those
    print(f"  Low-leakage (n={n}): before={avg_before:.4f} after={avg_after:.4f}")
    print(f"  Δ: {avg_after - avg_before:+.4f}")

    # Write results to supplemental file
    report_path = "reranker_results.md"
    with open(report_path, "w") as f:
        f.write(f"# ColBERT Reranker Test — Low-Leakage Conceptual Queries\n\n")
        f.write(f"Model: `answerdotai/answerai-colbert-small-v1` (dim=96)\n")
        f.write(f"Candidate pool: Combined (RRF k={RRF_K}) top-{RERANK_CUTOFF}\n")
        f.write(
            f"Queries: {n} low-leakage conceptual (overlap ≤ {100 * LEAKAGE_THRESHOLD:.0f}%)\n\n"
        )

        f.write(f"## Per-Query Results\n\n")
        f.write("| Q# | Leak% | C@1 | Col@1 | MRR before | MRR after | ΔMRR |\n")
        f.write("|----|-------|-----|-------|-----------|----------|------|\n")
        for r in sorted(low_leak_details, key=lambda x: x["qnum"]):
            gb = r["gold_rank_before"] if r["gold_rank_before"] else "miss"
            ga = r["gold_rank_after"] if r["gold_rank_after"] else "miss"
            f.write(
                f"| Q{r['qnum']} | {100 * r['leakage']:.0f}% | {gb} | {ga} | "
                f"{r['mrr_before']:.4f} | {r['mrr_after']:.4f} | {r['mrr_after'] - r['mrr_before']:+.4f} |\n"
            )

        f.write(f"\n## Summary\n\n")
        f.write(f"- Combined MRR before: {avg_before:.4f}\n")
        f.write(f"- ColBERT reranked MRR: {avg_after:.4f}\n")
        f.write(f"- ΔMRR: {avg_after - avg_before:+.4f}\n")
        f.write(f"- Improved: {improved}/{n}  Worsened: {worsened}/{n}\n")
        if n_reachable:
            f.write(
                f"- Gold in top-50 candidates: {n_reachable}/{n} "
                f"(reachable MRR before={reachable_before:.4f}, after={reachable_after:.4f})\n"
            )

    logger.info("Results written to %s", report_path)


if __name__ == "__main__":
    main()
