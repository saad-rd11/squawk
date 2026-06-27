"""
Precision eval at critical w_dense values to find the best compromise
between fixing heart attack and preserving EVB sparse save.
"""

import json
import logging
import statistics

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 200
RRF_K = 60

# The two adversarial cases
HA_QUERY = "passenger having a heart attack mid-flight paramedics met at the gate"
HA_TARGET = 1766047
HA_SPURIOUS = 1795879

EVB_QUERY = "EVB runway 7 taxiway E hold short lines faded"
EVB_TARGET = 1755782


def _collapse_to_parents(sp):
    o = {}
    for p in sp:
        rid = p.payload.get("report_id")
        if rid is not None:
            a = int(rid)
            if a not in o or p.score > o[a]:
                o[a] = p.score
    return sorted(o.items(), key=lambda x: x[1], reverse=True)


def _parent_mrr(collapsed, expected):
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn in expected:
            return 1.0 / rank
    return 0.0


def rrf(dense, sparse, wd, ws):
    dr = {a: r for r, (a, _) in enumerate(dense, 1)}
    sr = {a: r for r, (a, _) in enumerate(sparse, 1)}
    default = max(len(dense), len(sparse)) + 1
    scores = {}
    for a in set(dr) | set(sr):
        scores[a] = wd / (RRF_K + dr.get(a, default)) + ws / (
            RRF_K + sr.get(a, default)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )

    # Embed eval queries
    eval_texts = [q["query"] for q in queries]
    logger.info("Embedding %d eval queries...", len(eval_texts))
    eval_dvecs = ded.embed(eval_texts)
    eval_svecs = sed.embed(eval_texts)

    # Embed adversarial queries
    adv_texts = [HA_QUERY, EVB_QUERY]
    adv_dvecs = ded.embed(adv_texts)
    adv_svecs = sed.embed(adv_texts)

    mgr = Stage3Collection(path="./qdrant_storage")

    # Run all eval queries
    eval_results = []
    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        dense_raw = mgr.client.query_points(
            COLLECTION_NAME,
            query=eval_dvecs[i],
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        dense_collapsed = _collapse_to_parents(dense_raw)

        sparse_raw = mgr.client.query_points(
            COLLECTION_NAME,
            query=SparseVector(
                indices=list(eval_svecs[i].keys()), values=list(eval_svecs[i].values())
            ),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_collapsed = _collapse_to_parents(sparse_raw)

        eval_results.append(
            {
                "expected": expected,
                "dense": dense_collapsed,
                "sparse": sparse_collapsed,
            }
        )

    # Run adversarial queries
    ha_dense_raw = mgr.client.query_points(
        COLLECTION_NAME,
        query=adv_dvecs[0],
        using="dense",
        limit=SEARCH_LIMIT,
        with_payload=["report_id"],
    ).points
    ha_dense = _collapse_to_parents(ha_dense_raw)

    ha_sparse_raw = mgr.client.query_points(
        COLLECTION_NAME,
        query=SparseVector(
            indices=list(adv_svecs[0].keys()), values=list(adv_svecs[0].values())
        ),
        using="sparse",
        limit=SEARCH_LIMIT,
        with_payload=["report_id"],
    ).points
    ha_sparse = _collapse_to_parents(ha_sparse_raw)

    evb_dense_raw = mgr.client.query_points(
        COLLECTION_NAME,
        query=adv_dvecs[1],
        using="dense",
        limit=SEARCH_LIMIT,
        with_payload=["report_id"],
    ).points
    evb_dense = _collapse_to_parents(evb_dense_raw)

    evb_sparse_raw = mgr.client.query_points(
        COLLECTION_NAME,
        query=SparseVector(
            indices=list(adv_svecs[1].keys()), values=list(adv_svecs[1].values())
        ),
        using="sparse",
        limit=SEARCH_LIMIT,
        with_payload=["report_id"],
    ).points
    evb_sparse = _collapse_to_parents(evb_sparse_raw)

    mgr.close()

    # Test critical weights
    test_weights = [i / 20 for i in range(0, 21)]

    print(f"{'=' * 70}")
    print(f"  WEIGHT TRADE-OFF ANALYSIS")
    print(f"{'=' * 70}")
    print(
        f"  {'w_dense':>8}  {'w_sparse':>9}  {'Eval MRR':>9}  {'HA rank':>8}  {'HA sick?':>9}  {'EVB rank':>9}  {'EVB saved?':>10}"
    )
    print(
        f"  {'-' * 8}  {'-' * 9}  {'-' * 9}  {'-' * 8}  {'-' * 9}  {'-' * 9}  {'-' * 10}"
    )

    best_mrr = 0
    best_config = None

    for wd in test_weights:
        ws = 1.0 - wd

        # Eval MRR
        all_mrr = []
        for r in eval_results:
            combined = rrf(r["dense"], r["sparse"], wd, ws)
            mrr = _parent_mrr(combined, r["expected"])
            all_mrr.append(mrr)
        eval_mrr = statistics.mean(all_mrr)

        # Heart attack
        ha_combined = rrf(ha_dense, ha_sparse, wd, ws)
        ha_sick_rank = None
        for rank, (acn, _) in enumerate(ha_combined, 1):
            if acn == HA_TARGET:
                ha_sick_rank = rank
                break
        ha_wins = ha_sick_rank is not None and ha_sick_rank == 1

        # EVB
        evb_combined = rrf(evb_dense, evb_sparse, wd, ws)
        evb_target_rank = None
        for rank, (acn, _) in enumerate(evb_combined, 1):
            if acn == EVB_TARGET:
                evb_target_rank = rank
                break
        evb_saved = evb_target_rank is not None and evb_target_rank == 1

        ha_str = f"#{ha_sick_rank}" if ha_sick_rank else "miss"
        evb_str = f"#{evb_target_rank}" if evb_target_rank else "miss"

        marker = ""
        if eval_mrr > best_mrr:
            best_mrr = eval_mrr
            best_config = (wd, ws)
            marker = " <<< BEST MRR"

        # Highlight the two weights of interest
        if abs(wd - 0.70) < 0.01:
            marker += " <<< HA FIX"
        if abs(wd - 0.60) < 0.01:
            marker += " <<< EVB SAVE"

        print(
            f"  {wd:>8.2f}  {ws:>9.2f}  {eval_mrr:>9.4f}  {ha_str:>8}  {'YES' if ha_wins else 'no':>9}  {evb_str:>9}  {'YES' if evb_saved else 'no':>10}{marker}"
        )

    print()
    wd_best, ws_best = best_config
    print(
        f"  Best MRR: w_dense={wd_best:.2f}, w_sparse={ws_best:.2f} → MRR={best_mrr:.4f}"
    )
    print()
    print("  Trade-off summary:")
    print(f"    w_d <= 0.60: preserves EVB sparse save, but heart attack still broken")
    print(f"    w_d >= 0.70: fixes heart attack, but EVB drops to rank 2")
    print(f"    w_d = 0.65: compromises — neither, MRR intermediate")
    print(f"    w_d = 0.60: best MRR + EVB saved, but heart attack at rank 2")
    print(
        f"    w_d = 0.70: fixes heart attack at rank 1, EVB at rank 2, slightly lower MRR"
    )


if __name__ == "__main__":
    main()
