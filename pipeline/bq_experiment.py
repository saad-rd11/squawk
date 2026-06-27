"""
BQ + rescore simulation experiment.
Simulates Binary Quantization + rescore by:
1. Binarizing dense vectors to sign-based bits
2. Hamming distance search
3. Rescoring top-K with original cosine sim
4. RRF fusion with BM25 sparse
5. Compare MRR vs full-precision baseline
"""

import json
import logging
import time

import numpy as np
from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 200
RRF_K = 60
BQ_RESCORE_DEPTH = 50


def binarize(vec: np.ndarray) -> np.ndarray:
    return (vec > 0).astype(np.uint8)


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - (np.sum(a != b) / a.size)


def collapse_to_parent(scored: dict) -> list[tuple[int, float]]:
    """Max-score collapse across child points."""
    parent_scores = {}
    for acn, score in scored.items():
        if acn not in parent_scores or score > parent_scores[acn]:
            parent_scores[acn] = score
    return sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)


def compute_mrr(ranked, expected):
    for rank, (acn, _) in enumerate(ranked, 1):
        if acn in expected:
            return 1.0 / rank
    return 0.0


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    # ── Scroll ALL vectors from Qdrant ──
    mgr = Stage3Collection(path="./qdrant_storage")
    info = mgr.client.get_collection(COLLECTION_NAME)
    logger.info("Pool: %d points", info.points_count)

    points = []
    offset = None
    while True:
        page = mgr.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            offset=offset,
            with_payload=["report_id"],
            with_vectors=True,
        )
        points.extend(page[0])
        offset = page[1]
        if offset is None:
            break
    logger.info("Scrolled %d points", len(points))

    # Extract vectors and ACNs
    acn_vectors = {}  # acn -> (fp_vec, binary_vec)
    for pt in points:
        acn = int(pt.payload.get("report_id", 0))
        vec = np.array(
            pt.vector.get("dense") if isinstance(pt.vector, dict) else pt.vector,
            dtype=np.float32,
        )
        if acn not in acn_vectors or True:  # keep all child vectors
            if acn not in acn_vectors:
                acn_vectors[acn] = []
            acn_vectors[acn].append((vec, binarize(vec)))

    # ── Embed queries ──
    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )
    qtexts = [q["query"] for q in queries]
    logger.info("Embedding queries...")
    t0 = time.time()
    dvecs = ded.embed(qtexts)
    svecs = sed.embed(qtexts)
    logger.info("Embedded %d queries in %.1fs", len(qtexts), time.time() - t0)

    # ── Sparse search via Qdrant ──
    logger.info("Running sparse search...")
    sparse_results = []
    for i, svec in enumerate(svecs):
        raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(indices=list(svec.keys()), values=list(svec.values())),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_collapsed = {}
        for sp in raw:
            acn = int(sp.payload.get("report_id", 0))
            if acn not in sparse_collapsed or sp.score > sparse_collapsed[acn]:
                sparse_collapsed[acn] = sp.score
        sparse_results.append(
            sorted(sparse_collapsed.items(), key=lambda x: x[1], reverse=True)
        )
    mgr.close()

    # ── Per-query eval ──
    per_query = []
    default_rank = SEARCH_LIMIT + 1

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        qnum = i + 1
        qvec = np.array(dvecs[i], dtype=np.float32)
        qbin = binarize(qvec)

        # ── Full-precision cosine ──
        fp_scores = {}  # acn -> max score across child vecs
        for acn, vecs in acn_vectors.items():
            for fp_vec, _ in vecs:
                score = float(np.dot(qvec, fp_vec))
                if acn not in fp_scores or score > fp_scores[acn]:
                    fp_scores[acn] = score
        fp_ranked = collapse_to_parent(fp_scores)

        # ── BQ + rescore ──
        # Phase 1: Hamming distance search on all candidates
        bq_candidates = {}  # acn -> max hamming sim
        for acn, vecs in acn_vectors.items():
            for _, bvec in vecs:
                sim = hamming_similarity(qbin, bvec)
                if acn not in bq_candidates or sim > bq_candidates[acn]:
                    bq_candidates[acn] = sim
        # Phase 2: Rescore top-K with original cosine
        bq_top = sorted(bq_candidates.items(), key=lambda x: x[1], reverse=True)[
            :BQ_RESCORE_DEPTH
        ]
        bq_rescored = {}
        for acn, _ in bq_top:
            max_cos = max(np.dot(qvec, fp_vec) for fp_vec, _ in acn_vectors[acn])
            bq_rescored[acn] = float(max_cos)
        bq_ranked = sorted(bq_rescored.items(), key=lambda x: x[1], reverse=True)

        # ── RRF fusion with sparse ──
        def rrf_fuse(dense_ranked):
            dense_map = {acn: r for r, (acn, _) in enumerate(dense_ranked, 1)}
            sparse_map = {acn: r for r, (acn, _) in enumerate(sparse_results[i], 1)}
            all_acns = set(dense_map) | set(sparse_map)
            scores = {}
            for acn in all_acns:
                scores[acn] = 1.0 / (RRF_K + dense_map.get(acn, default_rank)) + 1.0 / (
                    RRF_K + sparse_map.get(acn, default_rank)
                )
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fp_combined = rrf_fuse(fp_ranked)
        bq_combined = rrf_fuse(bq_ranked)

        fp_mrr = compute_mrr(fp_combined, expected)
        bq_mrr = compute_mrr(bq_combined, expected)

        per_query.append(
            {
                "qnum": qnum,
                "query": q["query"][:60],
                "qtype": q.get("type", "conceptual"),
                "expected": list(expected),
                "fp_mrr": fp_mrr,
                "bq_mrr": bq_mrr,
                "delta": bq_mrr - fp_mrr,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info("Eval %d/%d", i + 1, len(queries))

    # ── Report ──
    n = len(per_query)
    fp_avg = sum(r["fp_mrr"] for r in per_query) / n
    bq_avg = sum(r["bq_mrr"] for r in per_query) / n
    n_improved = sum(1 for r in per_query if r["delta"] > 0)
    n_degraded = sum(1 for r in per_query if r["delta"] < 0)
    n_same = n - n_improved - n_degraded

    fp_hit1 = sum(1 for r in per_query if r["fp_mrr"] == 1.0)
    bq_hit1 = sum(1 for r in per_query if r["bq_mrr"] == 1.0)

    print()
    print("=" * 80)
    print("  BQ + RESCORE EXPERIMENT")
    print(
        f"  BQ fast search: Hamming distance  |  Rescore depth: top-{BQ_RESCORE_DEPTH}  |  RRF k={RRF_K}"
    )
    print("=" * 80)
    print()
    print(f"  {'':>25}  {'Full Prec':>10}  {'BQ+Rescore':>10}  {'Delta':>8}")
    print(f"  {'-' * 58}")
    print(
        f"  {'MRR (combined)':>25}  {fp_avg:.4f}  {bq_avg:.4f}  {bq_avg - fp_avg:+.4f}"
    )
    print(
        f"  {'Hit@1':>25}  {fp_hit1}/{n} ({100 * fp_hit1 / n:.1f}%)  {bq_hit1}/{n} ({100 * bq_hit1 / n:.1f}%)"
    )
    print(f"  {'Improved/Same/Degraded':>25}  {n_improved}/{n_same}/{n_degraded}")
    print()

    # By slice
    for label, key in [
        ("Low-leakage", "low_leakage"),
        ("High-leakage", "high_leakage"),
        ("Exact_token", "exact_token"),
        ("Conceptual", "conceptual"),
    ]:
        # approximate: conceptual = not exact_token
        if key == "exact_token":
            subset = [r for r in per_query if r["qtype"] == "exact_token"]
        elif key == "conceptual":
            subset = [r for r in per_query if r["qtype"] != "exact_token"]
        else:
            # low/high leakage: need leakage info — skip for now
            continue
        if not subset:
            continue
        sfp = sum(r["fp_mrr"] for r in subset) / len(subset)
        sbq = sum(r["bq_mrr"] for r in subset) / len(subset)
        print(
            f"  {f'MRR [{label}] ({len(subset)}q)':>25}  {sfp:.4f}  {sbq:.4f}  {sbq - sfp:+.4f}"
        )

    print()
    print("  ── Per-query detail (sorted by delta) ──")
    print(
        f"  {'Q':>3} {'Type':>6}  {'Full Prec':>10} {'BQ+Resc':>10} {'Delta':>8}  Query"
    )
    print(f"  {'-' * 65}")
    for r in sorted(per_query, key=lambda x: x["delta"]):
        marker = " ←" if r["delta"] < 0 else (" →" if r["delta"] > 0 else "")
        print(
            f"  {r['qnum']:>3} {r['qtype'][:6]:>6}  {r['fp_mrr']:.4f}  {r['bq_mrr']:.4f}  {r['delta']:+.4f}{marker}  {r['query']}"
        )

    print()
    print("=" * 80)
    print(f"  Full-precision MRR:  {fp_avg:.4f}")
    print(f"  BQ + rescore MRR:    {bq_avg:.4f}")
    print(f"  BQ degradation:      {bq_avg - fp_avg:+.4f}")
    print(f"  Memory savings:      ~32x compression on dense vectors")
    print("=" * 80)


if __name__ == "__main__":
    main()
