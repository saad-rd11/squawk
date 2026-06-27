"""
Compute comprehensive retrieval metrics for the current configuration.
Reports MRR, Recall@k, Precision@k, NDCG@k, MAP, Hit Rate, R-Precision
broken down by slice (overall, conceptual, exact_token, low-leak, high-leak)
and by search strategy (dense, sparse, combined).
"""

import json
import logging
import math
from statistics import mean

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KS = [1, 3, 5, 10, 20, 50]
SEARCH_LIMIT = 200
RRF_K = 60
LEAKAGE_THRESHOLD = 0.353
# Dense-weighted RRF: ratio 3:2 dense:sparse (conservative — preserves all sparse saves)
W_DENSE = 0.60
W_SPARSE = 0.40


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


def _rrf_combined(dense, sparse):
    dr = {acn: rank for rank, (acn, _) in enumerate(dense, 1)}
    sr = {acn: rank for rank, (acn, _) in enumerate(sparse, 1)}
    all_acns = set(dr) | set(sr)
    default = max(len(dense), len(sparse)) + 1
    scores = {}
    for acn in all_acns:
        scores[acn] = W_DENSE / (RRF_K + dr.get(acn, default)) + W_SPARSE / (
            RRF_K + sr.get(acn, default)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _parent_mrr(collapsed, expected):
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn in expected:
            return 1.0 / rank
    return 0.0


def _parent_recall(collapsed, expected, k):
    found = set(acn for acn, _ in collapsed[:k]) & expected
    return len(found) / len(expected) if expected else 0.0


def _parent_precision(collapsed, expected, k):
    topk = set(acn for acn, _ in collapsed[:k])
    return len(topk & expected) / k


def _dcg(relevances, k):
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def _parent_ndcg(collapsed, expected, k):
    relevance = [1 if acn in expected else 0 for acn, _ in collapsed[:k]]
    actual = _dcg(relevance, k)
    ideal_rel = sorted([1] * len(expected) + [0] * (k - len(expected)), reverse=True)[
        :k
    ]
    ideal = _dcg(ideal_rel, k)
    return actual / ideal if ideal > 0 else 0.0


def _parent_ap(collapsed, expected):
    hits = 0
    sum_prec = 0.0
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn in expected:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(expected) if expected else 0.0


def _parent_r_precision(collapsed, expected):
    r = len(expected)
    if r == 0:
        return 0.0
    found = set(acn for acn, _ in collapsed[:r]) & expected
    return len(found) / r


def compute_slice_metrics(results, label, n):
    if n == 0:
        return None
    out = {}
    for strat in ("dense", "sparse", "combined"):
        out[strat] = {}
        rr_vals = [r[f"{strat}_mrr"] for r in results]
        out[strat]["MRR"] = mean(rr_vals) if rr_vals else 0.0
        out[strat]["MAP"] = mean(r["ap"][strat] for r in results)
        out[strat]["R-Prec"] = mean(r["rprec"][strat] for r in results)
        out[strat]["Hit@1"] = (
            sum(1 for r in results if r[f"{strat}_mrr"] == 1.0) / n * 100
        )
        for k in KS:
            out[strat][f"Recall@{k}"] = (
                mean(r["recall"][strat][k] for r in results) * 100
            )
            out[strat][f"Prec@{k}"] = (
                mean(r["precision"][strat][k] for r in results) * 100
            )
            out[strat][f"NDCG@{k}"] = mean(r["ndcg"][strat][k] for r in results)
    return out


def print_strat_bar():
    print(f"  {'':>12}  {'Dense':>10}  {'Sparse':>10}  {'Combined':>10}")


def print_metric_row(label, vals, pct=False):
    fmt = "{:>7.2f}%" if pct else "{:>10.4f}"
    v1 = vals.get("dense", 0)
    v2 = vals.get("sparse", 0)
    v3 = vals.get("combined", 0)
    print(f"  {label:>12}  {fmt.format(v1)}  {fmt.format(v2)}  {fmt.format(v3)}")


def print_slice(table, slice_label, data):
    if data is None:
        return
    print(f"\n── {slice_label} ──\n")
    print_strat_bar()
    print("  " + "-" * 56)
    print_metric_row(
        "MRR", {s: data[s]["MRR"] for s in ("dense", "sparse", "combined")}
    )
    print_metric_row(
        "MAP", {s: data[s]["MAP"] for s in ("dense", "sparse", "combined")}
    )
    print_metric_row(
        "R-Prec", {s: data[s]["R-Prec"] for s in ("dense", "sparse", "combined")}
    )
    print_metric_row(
        "Hit@1",
        {s: data[s]["Hit@1"] for s in ("dense", "sparse", "combined")},
        pct=True,
    )
    print()
    for k in KS:
        print_metric_row(
            f"R@{k}",
            {s: data[s][f"Recall@{k}"] for s in ("dense", "sparse", "combined")},
            pct=True,
        )
    print()
    for k in KS:
        print_metric_row(
            f"P@{k}",
            {s: data[s][f"Prec@{k}"] for s in ("dense", "sparse", "combined")},
            pct=True,
        )
    print()
    for k in KS:
        print_metric_row(
            f"N@{k}", {s: data[s][f"NDCG@{k}"] for s in ("dense", "sparse", "combined")}
        )


def run():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))

    import pandas as pd

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)
    pool_df = df[df["ACN"].isin(pool_acns)].set_index("ACN")

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

    def compute_leakage(query_text, narrative_text):
        qtokens = set(query_text.lower().split())
        ntokens = set(narrative_text.lower().split())
        if not qtokens:
            return 0.0
        return len(qtokens & ntokens) / len(qtokens)

    per_query = []

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

        combined_collapsed = _rrf_combined(dense_collapsed, sparse_collapsed)

        # Leakage
        narrs = []
        for acn in expected:
            try:
                narrs.append(str(pool_df.loc[acn, "Narrative"]))
            except KeyError:
                pass
        full_narr = " ".join(narrs)
        leakage = compute_leakage(q["query"], full_narr)

        row = {
            "qnum": i + 1,
            "qtype": q.get("type", "conceptual"),
            "query": q["query"],
            "expected": expected,
            "leakage": leakage,
            "dense_mrr": _parent_mrr(dense_collapsed, expected),
            "sparse_mrr": _parent_mrr(sparse_collapsed, expected),
            "combined_mrr": _parent_mrr(combined_collapsed, expected),
            "recall": {},
            "precision": {},
            "ndcg": {},
            "ap": {},
            "rprec": {},
        }
        for strat, collapsed in [
            ("dense", dense_collapsed),
            ("sparse", sparse_collapsed),
            ("combined", combined_collapsed),
        ]:
            row["recall"][strat] = {}
            row["precision"][strat] = {}
            row["ndcg"][strat] = {}
            for k in KS:
                row["recall"][strat][k] = _parent_recall(collapsed, expected, k)
                row["precision"][strat][k] = _parent_precision(collapsed, expected, k)
                row["ndcg"][strat][k] = _parent_ndcg(collapsed, expected, k)
            row["ap"][strat] = _parent_ap(collapsed, expected)
            row["rprec"][strat] = _parent_r_precision(collapsed, expected)

        per_query.append(row)

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d", i + 1, len(queries))

    mgr.close()

    # Build slices
    all_q = per_query
    conceptual = [r for r in per_query if r["qtype"] == "conceptual"]
    exact_tok = [r for r in per_query if r["qtype"] == "exact_token"]
    low_leak = [r for r in conceptual if r["leakage"] <= LEAKAGE_THRESHOLD]
    high_leak = [r for r in conceptual if r["leakage"] > LEAKAGE_THRESHOLD]

    # =============================================================
    # OUTPUT
    # =============================================================
    print()
    print("=" * 70)
    print(f"  FULL METRICS REPORT — Current Config (anomaly + synopsis children)")
    print(
        f"  {len(per_query)} queries  /  {info.points_count} Qdrant pool points  /  RRF k={RRF_K}"
    )
    print("=" * 70)

    slices = [
        ("All queries (72)", all_q),
        ("Conceptual (57)", conceptual),
        ("  Low-leakage (15)", low_leak),
        ("  High-leakage (42)", high_leak),
        ("Exact_token (15)", exact_tok),
    ]

    for label, data in slices:
        metrics = compute_slice_metrics(data, label, len(data))
        print_slice(None, label, metrics)

    # ── Per-query table ──
    print()
    print("=" * 70)
    print("  PER-QUERY DETAIL")
    print("=" * 70)
    header = f"  {'Q':>3}  {'Type':>5}  {'Leak%':>5}  {'MRR-d':>6}  {'MRR-s':>6}  {'MRR-c':>6}  {'MAP-c':>6}  {'RPr-c':>6}  {'H@1?':>5}"
    print(header)
    print("  " + "-" * len(header))
    for r in per_query:
        hit = "✓" if r["combined_mrr"] == 1.0 else ""
        print(
            f"  {r['qnum']:>3}  {r['qtype'][:5]:>5}  {100 * r['leakage']:>4.0f}%  "
            f"{r['dense_mrr']:.4f}  {r['sparse_mrr']:.4f}  {r['combined_mrr']:.4f}  "
            f"{r['ap']['combined']:.4f}  {r['rprec']['combined']:.4f}  {hit:>5}"
        )

    # ── Distribution ──
    print()
    print("=" * 70)
    print("  COMBINED MRR DISTRIBUTION")
    print("=" * 70)
    mrr_vals = [r["combined_mrr"] for r in all_q]
    for threshold, label in [
        (1.0, "1.000 (perfect)"),
        (0.5, "0.500+"),
        (0.2, "0.200+"),
        (0.0, "> 0 (any)"),
    ]:
        cnt = sum(1 for v in mrr_vals if v >= threshold)
        print(
            f"  Queries with MRR >= {label}: {cnt:>2}/{len(all_q)} ({100 * cnt / len(all_q):.1f}%)"
        )

    # By rank
    rank_buckets = {"1": 0, "2-3": 0, "4-10": 0, "11-50": 0, "50+": 0, "miss": 0}
    # We don't have ranks directly in per_query, compute from MRR
    for r in all_q:
        m = r["combined_mrr"]
        if m == 1.0:
            rank_buckets["1"] += 1
        elif m >= 1 / 3:
            rank_buckets["2-3"] += 1
        elif m >= 0.1:
            rank_buckets["4-10"] += 1
        elif m >= 0.02:
            rank_buckets["11-50"] += 1
        elif m > 0:
            rank_buckets["50+"] += 1
        else:
            rank_buckets["miss"] += 1
    print()
    print(f"  Gold ACN rank distribution (combined):")
    for bucket, cnt in rank_buckets.items():
        print(f"    Rank {bucket}: {cnt:>2} ({100 * cnt / len(all_q):.1f}%)")

    print()
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    run()
