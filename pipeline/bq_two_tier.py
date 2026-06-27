"""
Two-tier quantization experiment:
  Tier 1: Binary Quantization (1-bit sign) for fast oversampling
  Tier 2: 4-bit TurboQuant-style rescoring
  Optional Tier 3: Full-precision rescoring

Measures the MRR tradeoff at each tier.
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


def binarize(vec: np.ndarray) -> np.ndarray:
    return (vec > 0).astype(np.uint8)


def quantize_4bit(
    vec: np.ndarray, dim_mins: np.ndarray, dim_maxs: np.ndarray
) -> np.ndarray:
    """Quantize vector to 4 bits per dimension using per-dim global min/max."""
    eps = 1e-8
    normalized = (vec - dim_mins) / (dim_maxs - dim_mins + eps)
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.round(normalized * 15).astype(np.uint8)


def dequantize_4bit(
    qvec: np.ndarray, dim_mins: np.ndarray, dim_maxs: np.ndarray
) -> np.ndarray:
    """Recover approximate float vector from 4-bit quantization."""
    eps = 1e-8
    return dim_mins + (qvec.astype(np.float32) / 15.0) * (dim_maxs - dim_mins + eps)


def collapse(scored: dict):
    parents = {}
    for acn, score in scored.items():
        if acn not in parents or score > parents[acn]:
            parents[acn] = score
    return sorted(parents.items(), key=lambda x: x[1], reverse=True)


def mrr(ranked, expected):
    for r, (acn, _) in enumerate(ranked, 1):
        if acn in expected:
            return 1.0 / r
    return 0.0


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    # ── Scroll vectors from Qdrant ──
    mgr = Stage3Collection(path="./qdrant_storage")
    info = mgr.client.get_collection(COLLECTION_NAME)
    logger.info("Pool: %d points", info.points_count)

    points = []
    offset = None
    while True:
        page = mgr.client.scroll(
            COLLECTION_NAME,
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

    # Build per-ACN vector lists
    acn_vecs = {}
    all_fp = []
    for pt in points:
        acn = int(pt.payload["report_id"])
        vec = np.array(
            pt.vector.get("dense") if isinstance(pt.vector, dict) else pt.vector,
            dtype=np.float32,
        )
        acn_vecs.setdefault(acn, []).append(vec)
        all_fp.append(vec)

    all_fp = np.array(all_fp)
    dim = all_fp.shape[1]
    logger.info(
        "Dim: %d, Total vecs: %d, Unique ACNs: %d", dim, len(all_fp), len(acn_vecs)
    )

    # ── Compute global per-dim min/max for 4-bit quantization ──
    dim_mins = all_fp.min(axis=0)
    dim_maxs = all_fp.max(axis=0)
    logger.info("4-bit quant: dim ranges computed")

    # ── Precompute binary and 4-bit vectors ──
    acn_bin = {}  # acn -> list of binary vectors
    acn_q4 = {}  # acn -> list of 4-bit quantized vectors
    for acn, vecs in acn_vecs.items():
        acn_bin[acn] = [binarize(v) for v in vecs]
        acn_q4[acn] = [quantize_4bit(v, dim_mins, dim_maxs) for v in vecs]

    # ── Embed queries ──
    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )
    qtexts = [q["query"] for q in queries]
    logger.info("Embedding queries...")
    dvecs = ded.embed(qtexts)
    svecs = sed.embed(qtexts)

    # ── Sparse search via Qdrant ──
    logger.info("Running sparse search...")
    sparse_results = []
    for svec in svecs:
        raw = mgr.client.query_points(
            COLLECTION_NAME,
            query=SparseVector(indices=list(svec.keys()), values=list(svec.values())),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sc = {}
        for sp in raw:
            a = int(sp.payload["report_id"])
            if a not in sc or sp.score > sc[a]:
                sc[a] = sp.score
        sparse_results.append(sorted(sc.items(), key=lambda x: x[1], reverse=True))
    mgr.close()

    # ── Strategy configs ──
    # Each strategy: (name, oversample_depth, rescore_type)
    strategies = [
        # (name, bq_depth, rescore_q4_depth, rescore_fp_depth)
        ("BQ only (no rescore)", 200, 0, 0),
        ("BQ → Q4@200", 200, 200, 0),
        ("BQ → Q4@100", 200, 100, 0),
        ("BQ → Q4@50", 200, 50, 0),
        ("BQ → Q4@30", 200, 30, 0),
        ("BQ → Q4@20", 200, 20, 0),
        ("BQ → Q4@10", 200, 10, 0),
        ("BQ → Q4@200 → FP@50", 200, 200, 50),
        ("BQ → Q4@200 → FP@20", 200, 200, 20),
        ("BQ → Q4@50 → FP@20", 200, 50, 20),
        ("BQ → Q4@20 → FP@20", 200, 20, 20),
        ("Full precision (FP only)", 200, 0, 200),
    ]

    default_rank = SEARCH_LIMIT + 1

    for name, bq_depth, q4_depth, fp_depth in strategies:
        mrr_vals = []
        hit1 = 0
        for i, q in enumerate(queries):
            expected = set(q["expected_acns"])
            qvec = np.array(dvecs[i], dtype=np.float32)
            qbin = binarize(qvec)

            # ── Tier 1: BQ oversampling ──
            bq_scores = {}
            for acn, bvecs in acn_bin.items():
                max_sim = max(1.0 - (np.sum(qbin != bv) / dim) for bv in bvecs)
                bq_scores[acn] = max_sim
            bq_top = sorted(bq_scores.items(), key=lambda x: x[1], reverse=True)[
                :bq_depth
            ]

            # ── Tier 2: 4-bit rescore (if enabled) ──
            if q4_depth > 0:
                q4_scores = {}
                for acn, _ in bq_top:
                    qvec_q4 = quantize_4bit(qvec, dim_mins, dim_maxs)
                    max_cos = max(
                        float(
                            np.dot(
                                dequantize_4bit(qv, dim_mins, dim_maxs),
                                dequantize_4bit(qvec_q4, dim_mins, dim_maxs),
                            )
                            / max(
                                np.linalg.norm(dequantize_4bit(qv, dim_mins, dim_maxs))
                                * np.linalg.norm(
                                    dequantize_4bit(qvec_q4, dim_mins, dim_maxs)
                                ),
                                1e-8,
                            )
                        )
                        for qv in acn_q4[acn]
                    )
                    q4_scores[acn] = max_cos
                tier2_ranked = sorted(
                    q4_scores.items(), key=lambda x: x[1], reverse=True
                )

                # Optional Tier 3: full-precision rescore
                if fp_depth > 0:
                    fp_scores = {}
                    tier2_candidates = tier2_ranked[:fp_depth]
                    for acn, _ in tier2_candidates:
                        max_cos = max(
                            float(
                                np.dot(qvec, fv)
                                / max(np.linalg.norm(qvec) * np.linalg.norm(fv), 1e-8)
                            )
                            for fv in acn_vecs[acn]
                        )
                        fp_scores[acn] = max_cos
                    dense_ranked = sorted(
                        fp_scores.items(), key=lambda x: x[1], reverse=True
                    )
                else:
                    dense_ranked = tier2_ranked

            elif fp_depth > 0:
                # Full precision directly (no BQ, no Q4)
                fp_scores = {}
                for acn, vecs in acn_vecs.items():
                    max_cos = max(
                        float(
                            np.dot(qvec, fv)
                            / max(np.linalg.norm(qvec) * np.linalg.norm(fv), 1e-8)
                        )
                        for fv in vecs
                    )
                    fp_scores[acn] = max_cos
                dense_ranked = sorted(
                    fp_scores.items(), key=lambda x: x[1], reverse=True
                )

            else:
                # BQ only, no rescore
                dense_ranked = bq_top[:bq_depth]

            # ── RRF with sparse ──
            dm = {acn: r for r, (acn, _) in enumerate(dense_ranked, 1)}
            sm = {acn: r for r, (acn, _) in enumerate(sparse_results[i], 1)}
            scores = {}
            for acn in set(dm) | set(sm):
                scores[acn] = 1.0 / (RRF_K + dm.get(acn, default_rank)) + 1.0 / (
                    RRF_K + sm.get(acn, default_rank)
                )
            combo = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            mr = mrr(combo, expected)
            mrr_vals.append(mr)
            if mr == 1.0:
                hit1 += 1

        avg_mrr = np.mean(mrr_vals)
        degraded = sum(1 for v in mrr_vals if v < mrr_vals[0]) if mrr_vals else 0
        print(
            f"  {name:>35s}: MRR={avg_mrr:.4f}  Hit@1={hit1}/72 ({100 * hit1 / 72:.1f}%)"
        )


if __name__ == "__main__":
    main()
