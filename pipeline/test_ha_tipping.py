"""
Find the exact w_dense tipping point for the heart attack query.
"""

import logging

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 200
RRF_K = 60
QUERY = "passenger having a heart attack mid-flight paramedics met at the gate"
SICK_ACN = 1766047
BATTERY_ACN = 1795879


def _collapse_to_parents(sp):
    o = {}
    for p in sp:
        rid = p.payload.get("report_id")
        if rid is not None:
            a = int(rid)
            if a not in o or p.score > o[a]:
                o[a] = p.score
    return sorted(o.items(), key=lambda x: x[1], reverse=True)


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
    mgr.close()

    print("Heart attack weight sweep:")
    print(
        f"  {'w_dense':>8}  {'w_sparse':>9}  {'Sick Rank':>10}  {'Battery Rank':>13}  {'Sick wins?':>10}"
    )
    for wd in [i / 20 for i in range(0, 21)]:
        ws = 1.0 - wd
        combined = rrf(dense_collapsed, sparse_collapsed, wd, ws)
        sick_rank = battery_rank = None
        for rank, (acn, _) in enumerate(combined, 1):
            if acn == SICK_ACN:
                sick_rank = rank
            if acn == BATTERY_ACN:
                battery_rank = rank
        wins = (
            "YES"
            if sick_rank and (battery_rank is None or sick_rank < battery_rank)
            else "no"
        )
        m = " <<< CONSERVATIVE" if abs(wd - 0.60) < 0.01 else ""
        print(
            f"  {wd:>8.2f}  {ws:>9.2f}  #{sick_rank:<7}  #{battery_rank:<10}  {wins:>5}{m}"
        )


if __name__ == "__main__":
    main()
