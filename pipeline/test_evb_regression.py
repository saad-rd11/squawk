"""
Deep-dive on Q21 EVB regression: what weight preserves the sparse save?
"""

import json
import logging

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 200
RRF_K = 60
QUERY = "EVB runway 7 taxiway E hold short lines faded"
TARGET = 1755782


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

    print("Q21 deep-dive: EVB runway 7 taxiway E hold short lines faded")
    print(f"Target ACN: {TARGET}")
    print()

    print("Dense top-5:")
    for rank, (acn, score) in enumerate(dense_collapsed[:5], 1):
        m = " <<< TARGET" if acn == TARGET else ""
        print(f"  #{rank}: ACN {acn} (score={score:.4f}){m}")

    print()
    print("Sparse top-5:")
    for rank, (acn, score) in enumerate(sparse_collapsed[:5], 1):
        m = " <<< TARGET" if acn == TARGET else ""
        print(f"  #{rank}: ACN {acn} (score={score:.4f}){m}")

    print()
    print("Weight sweep for this query:")
    print(f"  {'w_dense':>8}  {'w_sparse':>9}  {'Target Rank':>12}")
    for wd in [i / 20 for i in range(0, 21)]:
        ws = 1.0 - wd
        combined = rrf(dense_collapsed, sparse_collapsed, wd, ws)
        target_rank = None
        for rank, (acn, _) in enumerate(combined, 1):
            if acn == TARGET:
                target_rank = rank
                break
        marker = (
            " <<< CURRENT"
            if abs(wd - 0.80) < 0.01
            else (" <<< EQUAL" if abs(wd - 1.0) < 0.01 else "")
        )
        print(f"  {wd:>8.2f}  {ws:>9.2f}  #{target_rank:<5}{marker}")


if __name__ == "__main__":
    main()
