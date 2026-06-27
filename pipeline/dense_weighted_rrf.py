"""
Dense-weighted RRF fusion: find optimal dense:sparse weight ratio.

Experiments:
1. Global weight sweep (same weight for all query types)
2. Per-query-type weight sweep (different weights for conceptual vs exact_token)
3. Score-based fusion (normalized dense score + sparse score, not rank-based)
4. Dense-only fallback when sparse disagrees (conditional weighting)

Usage:
    ./venv/bin/python -m pipeline.dense_weighted_rrf
"""

import json
import logging
import math
import statistics
from collections import defaultdict

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 200
RRF_K = 60


def _collapse_to_parents(scored_points) -> list[tuple[int, float]]:
    acn_scores: dict[int, float] = {}
    for sp in scored_points:
        rid = sp.payload.get("report_id")
        if rid is not None:
            acn = int(rid)
            score = sp.score
            if acn not in acn_scores or score > acn_scores[acn]:
                acn_scores[acn] = score
    return sorted(acn_scores.items(), key=lambda x: x[1], reverse=True)


def _parent_mrr(collapsed: list[tuple[int, float]], expected: set[int]) -> float:
    for rank, (acn, _) in enumerate(collapsed, 1):
        if acn in expected:
            return 1.0 / rank
    return 0.0


def _parent_recall(
    collapsed: list[tuple[int, float]], expected: set[int], k: int
) -> float:
    found = set(acn for acn, _ in collapsed[:k]) & expected
    return len(found) / len(expected) if expected else 0.0


def rrf_standard(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    dr = {acn: rank for rank, (acn, _) in enumerate(dense, 1)}
    sr = {acn: rank for rank, (acn, _) in enumerate(sparse, 1)}
    all_acns = set(dr) | set(sr)
    default = max(len(dense), len(sparse)) + 1
    scores = {}
    for acn in all_acns:
        scores[acn] = 1.0 / (k + dr.get(acn, default)) + 1.0 / (
            k + sr.get(acn, default)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rrf_weighted(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    w_dense: float,
    w_sparse: float,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    dr = {acn: rank for rank, (acn, _) in enumerate(dense, 1)}
    sr = {acn: rank for rank, (acn, _) in enumerate(sparse, 1)}
    all_acns = set(dr) | set(sr)
    default = max(len(dense), len(sparse)) + 1
    scores = {}
    for acn in all_acns:
        scores[acn] = w_dense / (k + dr.get(acn, default)) + w_sparse / (
            k + sr.get(acn, default)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def score_fusion_normalized(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    w_dense: float = 0.5,
    w_sparse: float = 0.5,
) -> list[tuple[int, float]]:
    dm = dict(dense)
    sm = dict(sparse)
    all_acns = set(dm) | set(sm)

    d_scores = [s for _, s in dense]
    s_scores = [s for _, s in sparse]
    d_min, d_max = min(d_scores), max(d_scores)
    s_min, s_max = min(s_scores), max(s_scores)
    d_range = d_max - d_min if d_max > d_min else 1.0
    s_range = s_max - s_min if s_max > s_min else 1.0

    scores = {}
    for acn in all_acns:
        dn = (dm.get(acn, d_min) - d_min) / d_range
        sn = (sm.get(acn, s_min) - s_min) / s_range
        scores[acn] = w_dense * dn + w_sparse * sn
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def run_weight_sweep(all_results, label: str, weight_pairs: list[tuple], k=RRF_K):
    best = {"overall": -1, "config": None}
    best_exac = {"exac": -1, "config": None}
    best_conc = {"conc": -1, "config": None}
    rows = []

    for w_dense, w_sparse in weight_pairs:
        rr_all = []
        rr_exac = []
        rr_conc = []
        for r in all_results:
            collapsed = rrf_weighted(r["dense"], r["sparse"], w_dense, w_sparse, k)
            mrr = _parent_mrr(collapsed, r["expected"])
            rr_all.append(mrr)
            if r["qtype"] == "exact_token":
                rr_exac.append(mrr)
            else:
                rr_conc.append(mrr)

        mrr_overall = statistics.mean(rr_all)
        mrr_exac = statistics.mean(rr_exac) if rr_exac else 0
        mrr_conc = statistics.mean(rr_conc) if rr_conc else 0

        if mrr_overall > best["overall"]:
            best = {"overall": mrr_overall, "config": (w_dense, w_sparse)}
        if mrr_exac > best_exac["exac"]:
            best_exac = {"exac": mrr_exac, "config": (w_dense, w_sparse)}
        if mrr_conc > best_conc["conc"]:
            best_conc = {"conc": mrr_conc, "config": (w_dense, w_sparse)}

        rows.append((w_dense, w_sparse, mrr_overall, mrr_exac, mrr_conc))

    return rows, best, best_exac, best_conc


def result_table(rows, baseline_mrr, baseline_exac, baseline_conc, title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(
        f"  {'w_dense':>8}  {'w_sparse':>9}  {'MRR':>7}  {'Exac':>7}  {'Conc':>7}  {'ΔMRR':>7}  {'ΔExac':>7}  {'ΔConc':>7}"
    )
    print(
        f"  {'-' * 8}  {'-' * 9}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}"
    )
    for wd, ws, mrr, exac, conc in rows:
        dmrr = mrr - baseline_mrr
        dex = exac - baseline_exac
        dco = conc - baseline_conc
        marker = " <<<" if (exac >= baseline_exac and conc >= baseline_conc) else ""
        print(
            f"  {wd:>8.2f}  {ws:>9.2f}  {mrr:>7.4f}  {exac:>7.4f}  {conc:>7.4f}  {dmrr:>+7.4f}  {dex:>+7.4f}  {dco:>+7.4f}{marker}"
        )


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    has_types = all("type" in q for q in queries)
    n = len(queries)
    logger.info("Loaded %d queries (%s)", n, "typed" if has_types else "untyped")

    dense_embedder = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sparse_embedder = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )

    query_texts = [q["query"] for q in queries]
    logger.info("Embedding queries (dense)...")
    dense_vecs = dense_embedder.embed(query_texts)
    logger.info("Embedding queries (sparse)...")
    sparse_vecs = sparse_embedder.embed(query_texts)

    mgr = Stage3Collection(path="./qdrant_storage")
    info = mgr.client.get_collection(COLLECTION_NAME)
    logger.info("Collection: %d points", info.points_count)

    all_results = []
    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        dvec = dense_vecs[i]
        svec = sparse_vecs[i]
        qtype = q.get("type", "conceptual")

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

        all_results.append(
            {
                "i": i,
                "qtype": qtype,
                "expected": expected,
                "dense": dense_collapsed,
                "sparse": sparse_collapsed,
                "query": q["query"],
            }
        )

        if (i + 1) % 10 == 0:
            logger.info("  Queried %d/%d", i + 1, n)

    mgr.close()

    # --- BASELINE: standard equal-weight RRF ---
    baseline_rr = []
    baseline_rr_exac = []
    baseline_rr_conc = []
    for r in all_results:
        c = rrf_standard(r["dense"], r["sparse"])
        mrr = _parent_mrr(c, r["expected"])
        baseline_rr.append(mrr)
        if r["qtype"] == "exact_token":
            baseline_rr_exac.append(mrr)
        else:
            baseline_rr_conc.append(mrr)

    baseline_mrr = statistics.mean(baseline_rr)
    baseline_exac = statistics.mean(baseline_rr_exac) if baseline_rr_exac else 0
    baseline_conc = statistics.mean(baseline_rr_conc) if baseline_rr_conc else 0

    print()
    print("=" * 70)
    print("  BASELINE: Equal-weight RRF (w_dense=1.0, w_sparse=1.0)")
    print("=" * 70)
    print(f"    Overall MRR: {baseline_mrr:.4f}")
    print(f"    Exact_token: {baseline_exac:.4f}")
    print(f"    Conceptual:  {baseline_conc:.4f}")

    # ================================================================
    # EXPERIMENT 1: Global weight sweep (single weight for all queries)
    # ================================================================
    print()
    logger.info("Experiment 1: Global weight sweep...")
    weight_pairs = []
    for wd in [i / 100 for i in range(0, 110, 5)]:
        for ws in [i / 100 for i in range(0, 110, 5)]:
            if wd == 0 and ws == 0:
                continue
            weight_pairs.append((wd, ws))

    rows, best, best_exac, best_conc = run_weight_sweep(
        all_results, "Global", weight_pairs
    )

    result_table(
        rows,
        baseline_mrr,
        baseline_exac,
        baseline_conc,
        "EXPERIMENT 1: Global Weight Sweep",
    )

    wd_c, ws_c = best["config"]
    print(
        f"\n  BEST OVERALL: w_dense={wd_c:.2f}, w_sparse={ws_c:.2f} → MRR={best['overall']:.4f} (Δ={best['overall'] - baseline_mrr:+.4f})"
    )
    wd_c, ws_c = best_exac["config"]
    print(
        f"  BEST EXACT_TOKEN: w_dense={wd_c:.2f}, w_sparse={ws_c:.2f} → MRR={best_exac['exac']:.4f} (Δ={best_exac['exac'] - baseline_exac:+.4f})"
    )
    wd_c, ws_c = best_conc["config"]
    print(
        f"  BEST CONCEPTUAL: w_dense={wd_c:.2f}, w_sparse={ws_c:.2f} → MRR={best_conc['conc']:.4f} (Δ={best_conc['conc'] - baseline_conc:+.4f})"
    )

    # Find configs that beat baseline on BOTH slices
    both_winners = [
        (wd, ws, mrr, exac, conc)
        for wd, ws, mrr, exac, conc in rows
        if exac >= baseline_exac and conc >= baseline_conc
    ]
    if both_winners:
        both_winners.sort(key=lambda x: x[2], reverse=True)
        print(
            f"\n  Configs that beat baseline on BOTH slices ({len(both_winners)} found):"
        )
        for wd, ws, mrr, exac, conc in both_winners[:10]:
            print(
                f"    w_dense={wd:.2f}, w_sparse={ws:.2f}: overall={mrr:.4f} exac={exac:.4f} conc={conc:.4f}"
            )
    else:
        print(f"\n  No global weight beats baseline on both slices simultaneously.")

    # ================================================================
    # EXPERIMENT 2: Dense-only sweep (w_sparse=1.0 fixed, vary w_dense)
    # ================================================================
    print()
    logger.info("Experiment 2: Dense weight sweep (w_sparse=1.0 fixed)...")
    dense_sweep = [(wd, 1.0) for wd in [i / 10 for i in range(0, 31)]]
    rows2, best2, _, _ = run_weight_sweep(all_results, "Dense-sweep", dense_sweep)

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 2: Dense-only sweep (w_sparse=1.0 fixed)")
    print(f"{'=' * 70}")
    print(
        f"  {'w_dense':>8}  {'w_sparse':>9}  {'MRR':>7}  {'Exac':>7}  {'Conc':>7}  {'ΔMRR':>7}  {'ΔExac':>7}  {'ΔConc':>7}"
    )
    print(
        f"  {'-' * 8}  {'-' * 9}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}"
    )
    for wd, ws, mrr, exac, conc in rows2:
        dmrr = mrr - baseline_mrr
        dex = exac - baseline_exac
        dco = conc - baseline_conc
        marker = " <<<" if (exac >= baseline_exac and conc >= baseline_conc) else ""
        print(
            f"  {wd:>8.2f}  {ws:>9.2f}  {mrr:>7.4f}  {exac:>7.4f}  {conc:>7.4f}  {dmrr:>+7.4f}  {dex:>+7.4f}  {dco:>+7.4f}{marker}"
        )

    wd_c, ws_c = best2["config"]
    print(
        f"\n  BEST: w_dense={wd_c:.2f}, w_sparse={ws_c:.2f} → MRR={best2['overall']:.4f} (Δ={best2['overall'] - baseline_mrr:+.4f})"
    )

    # ================================================================
    # EXPERIMENT 3: Per-query-type weight sweep
    # ================================================================
    print()
    logger.info("Experiment 3: Per-query-type weight sweep...")

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 3: Per-query-type weights (w_dense_exac, w_dense_conc)")
    print(f"{'=' * 70}")

    best_pq = {"overall": -1, "config": None}
    pq_rows = []
    type_best_overall = {"overall": -1, "config": None, "exac_mrr": 0, "conc_mrr": 0}
    type_best_both = []

    for wde in [
        i / 20 for i in range(0, 21)
    ]:  # w_dense for exact_token: 0 to 1.0, step 0.05
        for wdc in [
            i / 20 for i in range(0, 21)
        ]:  # w_dense for conceptual: 0 to 1.0, step 0.05
            rr_all = []
            rr_exac = []
            rr_conc = []
            for r in all_results:
                wd = wde if r["qtype"] == "exact_token" else wdc
                ws = 1.0 - wd
                if ws < 0:
                    ws = 0.0
                collapsed = rrf_weighted(r["dense"], r["sparse"], wd, ws, RRF_K)
                mrr = _parent_mrr(collapsed, r["expected"])
                rr_all.append(mrr)
                if r["qtype"] == "exact_token":
                    rr_exac.append(mrr)
                else:
                    rr_conc.append(mrr)

            mrr_overall = statistics.mean(rr_all)
            mrr_exac = statistics.mean(rr_exac) if rr_exac else 0
            mrr_conc = statistics.mean(rr_conc) if rr_conc else 0

            if mrr_overall > type_best_overall["overall"]:
                type_best_overall = {
                    "overall": mrr_overall,
                    "config": (wde, wdc),
                    "exac_mrr": mrr_exac,
                    "conc_mrr": mrr_conc,
                }

            if mrr_exac >= baseline_exac and mrr_conc >= baseline_conc:
                type_best_both.append((wde, wdc, mrr_overall, mrr_exac, mrr_conc))

            pq_rows.append((wde, wdc, mrr_overall, mrr_exac, mrr_conc))

    type_best_both.sort(key=lambda x: x[2], reverse=True)

    print(
        f"  {'w_dexac':>8}  {'w_dconc':>8}  {'MRR':>7}  {'Exac':>7}  {'Conc':>7}  {'ΔMRR':>7}  {'ΔExac':>7}  {'ΔConc':>7}"
    )
    print(
        f"  {'-' * 8}  {'-' * 8}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}"
    )

    # Show interesting configs: those that beat baseline on both slices + neighbors
    shown = set()
    for wde, wdc, mrr, exac, conc in type_best_both[:15]:
        dmrr = mrr - baseline_mrr
        dex = exac - baseline_exac
        dco = conc - baseline_conc
        print(
            f"  {wde:>8.2f}  {wdc:>8.2f}  {mrr:>7.4f}  {exac:>7.4f}  {conc:>7.4f}  {dmrr:>+7.4f}  {dex:>+7.4f}  {dco:>+7.4f}  <<< BEATS BOTH"
        )
        shown.add((wde, wdc))

    wde_c, wdc_c = type_best_overall["config"]
    print(
        f"\n  BEST OVERALL: w_dense_exac={wde_c:.2f}, w_dense_conc={wdc_c:.2f} → MRR={type_best_overall['overall']:.4f} (Δ={type_best_overall['overall'] - baseline_mrr:+.4f})"
    )
    print(
        f"    Exac MRR: {type_best_overall['exac_mrr']:.4f} (Δ={type_best_overall['exac_mrr'] - baseline_exac:+.4f})"
    )
    print(
        f"    Conc MRR: {type_best_overall['conc_mrr']:.4f} (Δ={type_best_overall['conc_mrr'] - baseline_conc:+.4f})"
    )

    if type_best_both:
        wde, wdc, mrr, exac, conc = type_best_both[0]
        print(
            f"\n  BEST CONFIG BEATING BOTH SLICES: w_dense_exac={wde:.2f}, w_dense_conc={wdc:.2f}"
        )
        print(f"    Overall: {mrr:.4f}  Exac: {exac:.4f}  Conc: {conc:.4f}")

    # ================================================================
    # EXPERIMENT 4: Score-based fusion (normalized scores, not ranks)
    # ================================================================
    print()
    logger.info(
        "Experiment 4: Score-based fusion (normalized dense + sparse scores)..."
    )

    score_best = {"overall": -1, "config": None}
    score_rows = []
    for wd in [i / 20 for i in range(0, 21)]:
        ws = 1.0 - wd
        rr_all = []
        rr_exac = []
        rr_conc = []
        for r in all_results:
            collapsed = score_fusion_normalized(r["dense"], r["sparse"], wd, ws)
            mrr = _parent_mrr(collapsed, r["expected"])
            rr_all.append(mrr)
            if r["qtype"] == "exact_token":
                rr_exac.append(mrr)
            else:
                rr_conc.append(mrr)
        mrr_overall = statistics.mean(rr_all)
        mrr_exac = statistics.mean(rr_exac) if rr_exac else 0
        mrr_conc = statistics.mean(rr_conc) if rr_conc else 0
        score_rows.append((wd, ws, mrr_overall, mrr_exac, mrr_conc))
        if mrr_overall > score_best["overall"]:
            score_best = {"overall": mrr_overall, "config": (wd, ws)}

    result_table(
        score_rows,
        baseline_mrr,
        baseline_exac,
        baseline_conc,
        "EXPERIMENT 4: Score-based Fusion (min-max normalized)",
    )

    wd_c, ws_c = score_best["config"]
    print(
        f"\n  BEST: w_dense_score={wd_c:.2f}, w_sparse_score={ws_c:.2f} → MRR={score_best['overall']:.4f} (Δ={score_best['overall'] - baseline_mrr:+.4f})"
    )

    # ================================================================
    # EXPERIMENT 5: Conditional dense boost when sparse dominates
    # (heart-attack→battery fix: detect when sparse confidently ranks
    #  a wrong result and downweight it)
    # ================================================================
    print()
    logger.info(
        "Experiment 5: Conditional weighting (boost dense when dense & sparse disagree on top-1)..."
    )

    # For each query, if dense top-1 != sparse top-1, boost dense weight
    cond_best = {"overall": -1, "config": None}
    cond_rows = []
    for wd_normal in [i / 20 for i in range(4, 21)]:
        wd_normal_idx = int(round(wd_normal * 20))
        for wd_boost in [i / 20 for i in range(wd_normal_idx, 21)]:
            if wd_normal == 0:
                continue
            rr_all = []
            for r in all_results:
                dense_top1 = r["dense"][0][0] if r["dense"] else None
                sparse_top1 = r["sparse"][0][0] if r["sparse"] else None
                disagree = dense_top1 != sparse_top1
                wd = wd_boost if disagree else wd_normal
                ws = 1.0
                collapsed = rrf_weighted(r["dense"], r["sparse"], wd, ws, RRF_K)
                mrr = _parent_mrr(collapsed, r["expected"])
                rr_all.append(mrr)
            mrr_overall = statistics.mean(rr_all)
            cond_rows.append((wd_normal, wd_boost, mrr_overall))
            if mrr_overall > cond_best["overall"]:
                cond_best = {"overall": mrr_overall, "config": (wd_normal, wd_boost)}

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 5: Conditional dense boost (boost when top-1 disagree)")
    print(f"{'=' * 70}")
    print(f"  {'w_norm':>7}  {'w_boost':>8}  {'MRR':>7}  {'ΔMRR':>7}")
    for wdn, wdb, mrr in cond_rows:
        dmrr = mrr - baseline_mrr
        marker = " <<<" if mrr > baseline_mrr else ""
        if dmrr >= 0 or True:  # show all
            print(f"  {wdn:>7.2f}  {wdb:>8.2f}  {mrr:>7.4f}  {dmrr:>+7.4f}{marker}")

    wdn_c, wdb_c = cond_best["config"]
    print(
        f"\n  BEST: w_normal={wdn_c:.2f}, w_boost={wdb_c:.2f} → MRR={cond_best['overall']:.4f} (Δ={cond_best['overall'] - baseline_mrr:+.4f})"
    )

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(
        f"  Baseline (equal weight): MRR={baseline_mrr:.4f}  Exac={baseline_exac:.4f}  Conc={baseline_conc:.4f}"
    )
    print()

    # Best global
    wd, ws = best["config"]
    print(
        f"  Best global weight:              w_d={wd:.2f} w_s={ws:.2f} → MRR={best['overall']:.4f} (Δ={best['overall'] - baseline_mrr:+.4f})"
    )
    wd, ws = best_exac["config"]
    print(
        f"  Best for exact_token:            w_d={wd:.2f} w_s={ws:.2f} → MRR={best_exac['exac']:.4f} (Δ={best_exac['exac'] - baseline_exac:+.4f})"
    )
    wd, ws = best_conc["config"]
    print(
        f"  Best for conceptual:             w_d={wd:.2f} w_s={ws:.2f} → MRR={best_conc['conc']:.4f} (Δ={best_conc['conc'] - baseline_conc:+.4f})"
    )

    if type_best_both:
        wde, wdc, mrr, exac, conc = type_best_both[0]
        print(
            f"  Best per-type (both slices):      w_dex={wde:.2f} w_dcon={wdc:.2f} → MRR={mrr:.4f} Exac={exac:.4f} Conc={conc:.4f}"
        )

    wde_c, wdc_c = type_best_overall["config"]
    print(
        f"  Best per-type (overall):          w_dex={wde_c:.2f} w_dcon={wdc_c:.2f} → MRR={type_best_overall['overall']:.4f}"
    )

    wd_c, ws_c = score_best["config"]
    print(
        f"  Best score-based:                 w_d={wd_c:.2f} w_s={ws_c:.2f} → MRR={score_best['overall']:.4f} (Δ={score_best['overall'] - baseline_mrr:+.4f})"
    )

    wdn_c, wdb_c = cond_best["config"]
    print(
        f"  Best conditional boost:           w_norm={wdn_c:.2f} w_boost={wdb_c:.2f} → MRR={cond_best['overall']:.4f} (Δ={cond_best['overall'] - baseline_mrr:+.4f})"
    )

    print()
    print(f"  Total weight configs tested (E1): {len(weight_pairs)}")
    print(f"  Total weight configs tested (E2): {len(dense_sweep)}")
    print(f"  Total weight configs tested (E3): {len(pq_rows)}")
    print(f"  Total weight configs tested (E4): {len(score_rows)}")
    print(f"  Total weight configs tested (E5): {len(cond_rows)}")


if __name__ == "__main__":
    main()
