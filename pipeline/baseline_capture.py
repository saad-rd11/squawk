"""
Baseline: run identical eval as verify_findings, save per-query results as JSON.

Usage:
    ./venv/bin/python -m pipeline.baseline_capture
"""

import json
import logging
from collections import defaultdict

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 300
RRF_K_VALUES = [5, 10, 20, 30, 40, 60, 80]


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
    qtokens = set(query_text.lower().split())
    ntokens = set(narrative_text.lower().split())
    if not qtokens:
        return 0.0
    return len(qtokens & ntokens) / len(qtokens)


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    import pandas as pd

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)

    # Load pool ACNs for leakage computation
    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))
    pool_df = df[df["ACN"].isin(pool_acns)].set_index("ACN")

    # Embed queries
    dense_embedder = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sparse_embedder = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )

    mgr = Stage3Collection(path="./qdrant_storage")
    info = mgr.client.get_collection(COLLECTION_NAME)
    logger.info("Collection: %d points", info.points_count)

    query_texts = [q["query"] for q in queries]
    logger.info("Embedding queries...")
    dense_vecs = dense_embedder.embed(query_texts)
    sparse_vecs = sparse_embedder.embed(query_texts)
    logger.info("Done embedding")

    all_results = []
    default_rank = SEARCH_LIMIT + 1

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
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

        combined_ranks_by_k = {}
        all_acns = set(dense_ranks) | set(sparse_ranks)

        for k in RRF_K_VALUES:
            scores = {}
            for acn in all_acns:
                scores[acn] = 1.0 / (k + dense_ranks.get(acn, default_rank)) + 1.0 / (
                    k + sparse_ranks.get(acn, default_rank)
                )
            combined_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            combined_ranks_by_k[k] = {
                acn: rank for rank, (acn, _) in enumerate(combined_sorted, 1)
            }

        gold_acn = next(iter(expected), None)

        drr = _parent_mrr(dense_collapsed, expected)
        srr = _parent_mrr(sparse_collapsed, expected)
        crr_by_k = {}
        for k in RRF_K_VALUES:
            sorted_c = sorted(combined_ranks_by_k[k].items(), key=lambda x: x[1])
            crr_by_k[k] = _parent_mrr([(acn, 0) for acn, _ in sorted_c], expected)

        # Leakage
        narrs = []
        for acn in expected:
            try:
                narrs.append(str(pool_df.loc[acn, "Narrative"]))
            except KeyError:
                pass
        full_narr = " ".join(narrs)
        leakage = compute_leakage(q["query"], full_narr)

        # Top-10 collapsed for false-positive audit
        combined_sorted_k60 = sorted(
            combined_ranks_by_k[60].items(), key=lambda x: x[1]
        )
        top10_combined = [
            {"acn": acn, "rank": rank} for acn, rank in combined_sorted_k60[:10]
        ]
        top10_dense = [
            {"acn": acn, "score": score} for acn, score in dense_collapsed[:10]
        ]
        top10_sparse = [
            {"acn": acn, "score": score} for acn, score in sparse_collapsed[:10]
        ]

        all_results.append(
            {
                "i": i,
                "q_num": i + 1,
                "query": q["query"],
                "qtype": q.get("type", "conceptual"),
                "expected": list(expected),
                "dense_gold_rank": dense_ranks.get(gold_acn, default_rank)
                if gold_acn
                else default_rank,
                "sparse_gold_rank": sparse_ranks.get(gold_acn, default_rank)
                if gold_acn
                else default_rank,
                "combined_gold_ranks": {
                    k: ranks.get(gold_acn, default_rank) if gold_acn else default_rank
                    for k, ranks in combined_ranks_by_k.items()
                },
                "drr": drr,
                "srr": srr,
                "crr_by_k": crr_by_k,
                "leakage": leakage,
                "n_expected": len(expected),
                "top10_dense": top10_dense,
                "top10_sparse": top10_sparse,
                "top10_combined": top10_combined,
            }
        )

    mgr.close()

    baseline = {
        "point_count": info.points_count,
        "n_queries": len(all_results),
        "results": all_results,
    }

    with open("baseline_results.json", "w") as f:
        json.dump(baseline, f, indent=2)

    logger.info(
        "Baseline saved to baseline_results.json (%d queries, %d points)",
        len(all_results),
        info.points_count,
    )


if __name__ == "__main__":
    main()
