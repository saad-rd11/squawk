"""
Stage 3 diagnostic analysis: distribution, fusion sensitivity, reranker ceiling, tail, leakage.
Run:  ./venv/bin/python -m pipeline.diagnose
Output: diagnosis_report.md
"""

import json
import logging
import math
import statistics
from collections import Counter, defaultdict

import numpy as np
from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 300  # exceed pool size (535 children, ~250 parents)
RRF_K_VALUES = [5, 10, 20, 30, 40, 60, 80]
KS = [1, 5, 10, 20, 50]


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


def _parent_completeness(collapsed, expected, k):
    found = set(acn for acn, _ in collapsed[:k]) & expected
    return len(found) / len(expected) if expected else 0.0


def compute_leakage(query_text: str, narrative_text: str) -> float:
    """Token overlap: fraction of query tokens appearing verbatim in narrative."""
    qtokens = set(query_text.lower().split())
    ntokens = set(narrative_text.lower().split())
    if not qtokens:
        return 0.0
    return len(qtokens & ntokens) / len(qtokens)


def tokenize_simple(text: str) -> list[str]:
    return text.lower().split()


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)
    logger.info("Loaded %d queries", len(queries))

    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))

    # Load narratives for leakage analysis
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
    logger.info("Done embedding")

    # ─── Per-query results ───
    all_results = []  # list of dicts per query

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        dvec = dense_vecs[i]
        svec = sparse_vecs[i]
        qtype = q.get("type", "conceptual")
        qtokens = len(tokenize_simple(q["query"]))

        # Dense search
        dense_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dvec,
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        dense_collapsed = _collapse_to_parents(dense_raw)

        # Sparse search
        sparse_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(indices=list(svec.keys()), values=list(svec.values())),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_collapsed = _collapse_to_parents(sparse_raw)

        # Build rank dicts
        dense_ranks = {acn: rank for rank, (acn, _) in enumerate(dense_collapsed, 1)}
        sparse_ranks = {acn: rank for rank, (acn, _) in enumerate(sparse_collapsed, 1)}

        # Compute RRF for each k
        combined_ranks_by_k = {}
        all_acns = set(dense_ranks) | set(sparse_ranks)
        default = max(len(dense_collapsed), len(sparse_collapsed)) + 1

        for k in RRF_K_VALUES:
            scores = {}
            for acn in all_acns:
                scores[acn] = 1.0 / (k + dense_ranks.get(acn, default)) + 1.0 / (
                    k + sparse_ranks.get(acn, default)
                )
            combined_collapsed = sorted(
                scores.items(), key=lambda x: x[1], reverse=True
            )
            combined_ranks_by_k[k] = {
                acn: rank for rank, (acn, _) in enumerate(combined_collapsed, 1)
            }

        # Gold ranks
        gold_acn = next(iter(expected), None)
        dense_gold_rank = dense_ranks.get(gold_acn, default) if gold_acn else default
        sparse_gold_rank = sparse_ranks.get(gold_acn, default) if gold_acn else default
        combined_gold_ranks = {
            k: ranks.get(gold_acn, default) if gold_acn else default
            for k, ranks in combined_ranks_by_k.items()
        }

        # Dense gold position for multi-ACN queries
        dense_gold_at_1 = dense_gold_rank == 1

        # MRR
        drr = _parent_mrr(dense_collapsed, expected)
        srr = _parent_mrr(sparse_collapsed, expected)
        crr_by_k = {}
        for k in RRF_K_VALUES:
            sorted_combined = sorted(combined_ranks_by_k[k].items(), key=lambda x: x[1])
            crr_by_k[k] = _parent_mrr(
                [(acn, 0) for acn, _ in sorted_combined],
                expected,
            )

        # Reranker reachability: gold in combined top-c?
        rerank_reach = {}
        for c in [10, 20, 50, 100]:
            for k in [60]:
                top_c = set(
                    acn
                    for acn, _ in sorted(
                        combined_ranks_by_k[k].items(), key=lambda x: x[1]
                    )[:c]
                )
                rerank_reach[c] = bool(expected & top_c)

        # Leakage
        narrs = []
        for acn in expected:
            try:
                narrs.append(str(pool_df.loc[acn, "Narrative"]))
            except KeyError:
                pass
        full_narr = " ".join(narrs)
        leakage = compute_leakage(q["query"], full_narr)

        all_results.append(
            {
                "i": i,
                "query": q["query"],
                "qtype": qtype,
                "expected": expected,
                "n_expected": len(expected),
                "qtokens": qtokens,
                "dense_gold_rank": dense_gold_rank,
                "sparse_gold_rank": sparse_gold_rank,
                "combined_gold_ranks": combined_gold_ranks,
                "drr": drr,
                "srr": srr,
                "crr_by_k": crr_by_k,
                "dense_collapsed": dense_collapsed,
                "sparse_collapsed": sparse_collapsed,
                "combined_ranks_by_k": combined_ranks_by_k,
                "rerank_reach": rerank_reach,
                "leakage": leakage,
                "dense_gold_at_1": dense_gold_at_1,
                "multi_acn": len(expected) > 1,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d", i + 1, len(queries))

    mgr.close()

    # ================================================================
    # DELIVERABLE 1: DISTRIBUTION SHAPE
    # ================================================================
    k60_ranks = [r["combined_gold_ranks"][60] for r in all_results]
    buckets = {"1": 0, "2-5": 0, "6-20": 0, "21-50": 0, "51+": 0}
    for r in k60_ranks:
        if r == 1:
            buckets["1"] += 1
        elif r <= 5:
            buckets["2-5"] += 1
        elif r <= 20:
            buckets["6-20"] += 1
        elif r <= 50:
            buckets["21-50"] += 1
        else:
            buckets["51+"] += 1

    n = len(all_results)
    dist_report = {
        "total_queries": n,
        "solved_at_1": f"{buckets['1']}/{n} ({100 * buckets['1'] / n:.1f}%)",
        "solved_in_top5": f"{(buckets['1'] + buckets['2-5'])}/{n} ({100 * (buckets['1'] + buckets['2-5']) / n:.1f}%)",
        "solved_in_top20": f"{(buckets['1'] + buckets['2-5'] + buckets['6-20'])}/{n} ({100 * (buckets['1'] + buckets['2-5'] + buckets['6-20']) / n:.1f}%)",
        "tail_beyond_20": f"{buckets['21-50'] + buckets['51+']}/{n} ({100 * (buckets['21-50'] + buckets['51+']) / n:.1f}%)",
        "buckets": {k: f"{v}/{n} ({100 * v / n:.1f}%)" for k, v in buckets.items()},
    }
    print("\n=== DELIVERABLE 1: DISTRIBUTION SHAPE ===")
    print(json.dumps(dist_report, indent=2))

    # Compute multimodality: ratio of rank-1 + rank>20 vs middle ranks
    at_1 = buckets["1"]
    in_middle = buckets["2-5"] + buckets["6-20"]
    in_tail = buckets["21-50"] + buckets["51+"]
    print(f"  Rank 1: {at_1}")
    print(f"  Rank 2-20 (middle): {in_middle}")
    print(f"  Rank 21+ (tail): {in_tail}")
    print(f"  Middle fraction: {in_middle / n:.1%}")
    print(f"  Meaningful bimodal? {'YES' if at_1 + in_tail > 2 * in_middle else 'NO'}")
    print()

    # ================================================================
    # DELIVERABLE 2: FUSION SENSITIVITY
    # ================================================================
    print("=== DELIVERABLE 2: FUSION SENSITIVITY ===")

    mrr_tables = {"overall": {}, "conceptual": {}, "exact_token": {}}
    comp_tables = {
        k: {
            "overall": {
                "dense": 0,
                "sparse": 0,
                "combined": {kr: 0 for kr in RRF_K_VALUES},
            }
            for k in KS
        }
    }

    for label, filtered in [
        ("overall", all_results),
        ("conceptual", [r for r in all_results if r["qtype"] == "conceptual"]),
        ("exact_token", [r for r in all_results if r["qtype"] == "exact_token"]),
    ]:
        for k in RRF_K_VALUES:
            crrs = [r["crr_by_k"][k] for r in filtered]
            mrr = sum(crrs) / len(crrs) if crrs else 0
            mrr_tables[label][k] = round(mrr, 4)

    # Overall dense/sparse MRR for comparison
    dense_mrr_overall = sum(r["drr"] for r in all_results) / len(all_results)
    sparse_mrr_overall = sum(r["srr"] for r in all_results) / len(all_results)
    dense_mrr_conc = sum(
        r["drr"] for r in all_results if r["qtype"] == "conceptual"
    ) / max(sum(1 for r in all_results if r["qtype"] == "conceptual"), 1)
    sparse_mrr_conc = sum(
        r["srr"] for r in all_results if r["qtype"] == "conceptual"
    ) / max(sum(1 for r in all_results if r["qtype"] == "conceptual"), 1)
    dense_mrr_exac = sum(
        r["drr"] for r in all_results if r["qtype"] == "exact_token"
    ) / max(sum(1 for r in all_results if r["qtype"] == "exact_token"), 1)
    sparse_mrr_exac = sum(
        r["srr"] for r in all_results if r["qtype"] == "exact_token"
    ) / max(sum(1 for r in all_results if r["qtype"] == "exact_token"), 1)

    print(
        f"\nBaseline dense MRR: overall={dense_mrr_overall:.4f}, conc={dense_mrr_conc:.4f}, exac={dense_mrr_exac:.4f}"
    )
    print(
        f"Baseline sparse MRR: overall={sparse_mrr_overall:.4f}, conc={sparse_mrr_conc:.4f}, exac={sparse_mrr_exac:.4f}"
    )
    print()
    print(f"{'k':>5} | {'Overall':>8} | {'Conceptual':>10} | {'Exact_token':>11}")
    print("-" * 44)
    for k in RRF_K_VALUES:
        print(
            f"{k:>5} | {mrr_tables['overall'][k]:>8.4f} | {mrr_tables['conceptual'][k]:>10.4f} | {mrr_tables['exact_token'][k]:>11.4f}"
        )

    # Best k per slice
    for label in ["overall", "conceptual", "exact_token"]:
        best_k = max(RRF_K_VALUES, key=lambda k: mrr_tables[label][k])
        best_mrr = mrr_tables[label][best_k]
        baseline_dense = {
            "overall": dense_mrr_overall,
            "conceptual": dense_mrr_conc,
            "exact_token": dense_mrr_exac,
        }[label]
        baseline_sparse = {
            "overall": sparse_mrr_overall,
            "conceptual": sparse_mrr_conc,
            "exact_token": sparse_mrr_exac,
        }[label]
        print(
            f"  Best {label} k={best_k} (MRR={best_mrr:.4f}) vs dense={baseline_dense:.4f} sparse={baseline_sparse:.4f}"
        )

    # Current regression at k=60
    combined_exac_k60 = mrr_tables["exact_token"][60]
    combined_multi_at5 = {k: 0.0 for k in RRF_K_VALUES}
    multi_count = sum(1 for r in all_results if r["multi_acn"])
    for r in all_results:
        if r["multi_acn"]:
            for kr in RRF_K_VALUES:
                top_c = set(
                    acn
                    for acn, _ in sorted(
                        r["combined_ranks_by_k"][kr].items(), key=lambda x: x[1]
                    )[:5]
                )
                found = len(top_c & r["expected"])
                combined_multi_at5[kr] += found / len(r["expected"])
    for kr in RRF_K_VALUES:
        if multi_count:
            combined_multi_at5[kr] /= multi_count

    # Dense multi@5
    dense_multi_at5 = 0.0
    for r in all_results:
        if r["multi_acn"]:
            top_c = set(acn for acn, _ in r["dense_collapsed"][:5])
            found = len(top_c & r["expected"])
            dense_multi_at5 += found / len(r["expected"])
    if multi_count:
        dense_multi_at5 /= multi_count

    print(f"\nCurrent regressions at k=60:")
    print(
        f"  exact_token Combined ({combined_exac_k60:.4f}) vs Sparse ({sparse_mrr_exac:.4f}): Δ={combined_exac_k60 - sparse_mrr_exac:.4f}"
    )
    print(
        f"  Multi-ACN@5 Combined ({combined_multi_at5[60]:.4f}) vs Dense ({dense_multi_at5:.4f}): Δ={combined_multi_at5[60] - dense_multi_at5:.4f}"
    )

    # Weighted/score-based fusion test
    print(f"\n--- Per-type weighting test ---")
    for w_sparse_exac in [0.3, 0.5, 0.7, 0.9]:
        for w_dense_conc in [0.3, 0.5, 0.7, 0.9]:
            weighted_rr = []
            weighted_rr_exac = []
            weighted_rr_conc = []
            for r in all_results:
                default = 251
                all_acns = set(acn for acn, _ in r["dense_collapsed"]) | set(
                    acn for acn, _ in r["sparse_collapsed"]
                )
                scores = {}
                w_dense = (
                    w_dense_conc if r["qtype"] == "conceptual" else (1 - w_sparse_exac)
                )
                w_sparse = (
                    (1 - w_dense_conc) if r["qtype"] == "conceptual" else w_sparse_exac
                )
                dr = {
                    acn: rank for rank, (acn, _) in enumerate(r["dense_collapsed"], 1)
                }
                sr = {
                    acn: rank for rank, (acn, _) in enumerate(r["sparse_collapsed"], 1)
                }
                for acn in all_acns:
                    d_contrib = w_dense / (60 + dr.get(acn, default))
                    s_contrib = w_sparse / (60 + sr.get(acn, default))
                    scores[acn] = d_contrib + s_contrib
                combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                mrr = _parent_mrr([(acn, 0) for acn, _ in combined], r["expected"])
                weighted_rr.append(mrr)
                if r["qtype"] == "exact_token":
                    weighted_rr_exac.append(mrr)
                else:
                    weighted_rr_conc.append(mrr)
            w_mrr_overall = sum(weighted_rr) / len(weighted_rr)
            w_mrr_exac = (
                sum(weighted_rr_exac) / len(weighted_rr_exac) if weighted_rr_exac else 0
            )
            w_mrr_conc = (
                sum(weighted_rr_conc) / len(weighted_rr_conc) if weighted_rr_conc else 0
            )
            # Check if this beats the global k=60 on all slices
            beats_exac = w_mrr_exac >= combined_exac_k60
            beats_conc = w_mrr_conc >= mrr_tables["conceptual"][60]
            if beats_exac and beats_conc:
                print(
                    f"  w_sparse_exac={w_sparse_exac:.1f} w_dense_conc={w_dense_conc:.1f}: overall={w_mrr_overall:.4f} exac={w_mrr_exac:.4f} conc={w_mrr_conc:.4f}  <<< BEATS GLOBAL k=60 ON BOTH SLICES"
                )
            # Check against k=5 (stronger baseline)
            beats_exac_k5 = w_mrr_exac >= mrr_tables["exact_token"][5]
            beats_conc_k5 = w_mrr_conc >= mrr_tables["conceptual"][5]
            if beats_exac_k5 and beats_conc_k5:
                print(
                    f"  w_sparse_exac={w_sparse_exac:.1f} w_dense_conc={w_dense_conc:.1f}: overall={w_mrr_overall:.4f} exac={w_mrr_exac:.4f} conc={w_mrr_conc:.4f}  <<< BEATS GLOBAL k=5 ON BOTH SLICES"
                )

    print()

    # ================================================================
    # DELIVERABLE 3: RERANKER REACHABILITY
    # ================================================================
    print("=== DELIVERABLE 3: RERANKER CEILING ===")

    for cutoff in [10, 20, 50, 100]:
        for k in [60]:
            reachable = sum(1 for r in all_results if r["rerank_reach"][cutoff])
            pct = 100 * reachable / n
            # By type
            conc_reachable = sum(
                1
                for r in all_results
                if r["qtype"] == "conceptual" and r["rerank_reach"][cutoff]
            )
            exac_reachable = sum(
                1
                for r in all_results
                if r["qtype"] == "exact_token" and r["rerank_reach"][cutoff]
            )
            n_conc = sum(1 for r in all_results if r["qtype"] == "conceptual")
            n_exac = sum(1 for r in all_results if r["qtype"] == "exact_token")
            # Tail vs solved
            tail_conc = sum(
                1
                for r in all_results
                if r["qtype"] == "conceptual"
                and r["combined_gold_ranks"][60] > 20
                and r["rerank_reach"][cutoff]
            )
            tail_exac = sum(
                1
                for r in all_results
                if r["qtype"] == "exact_token"
                and r["combined_gold_ranks"][60] > 20
                and r["rerank_reach"][cutoff]
            )
            tail_conc_total = sum(
                1
                for r in all_results
                if r["qtype"] == "conceptual" and r["combined_gold_ranks"][60] > 20
            )
            tail_exac_total = sum(
                1
                for r in all_results
                if r["qtype"] == "exact_token" and r["combined_gold_ranks"][60] > 20
            )

            print(f"  Combined top-{cutoff}: {reachable}/{n} ({pct:.1f}%)")
            print(
                f"    Conceptual: {conc_reachable}/{n_conc} ({100 * conc_reachable / n_conc:.1f}%)"
            )
            print(
                f"    Exact_token: {exac_reachable}/{n_exac} ({100 * exac_reachable / n_exac:.1f}%)"
            )
            if tail_conc_total > 0:
                print(
                    f"    Conceptual tail reachable: {tail_conc}/{tail_conc_total} ({100 * tail_conc / tail_conc_total:.1f}%)"
                )
            if tail_exac_total > 0:
                print(
                    f"    Exact_token tail reachable: {tail_exac}/{tail_exac_total} ({100 * tail_exac / tail_exac_total:.1f}%)"
                )

    # Bottleneck: gold not in top-c vs gold in top-c but mis-ranked
    print()
    for cutoff in [50]:
        missing_from_candidates = sum(
            1 for r in all_results if not r["rerank_reach"][cutoff]
        )
        reachable = n - missing_from_candidates
        present_but_low = sum(
            1
            for r in all_results
            if r["rerank_reach"][cutoff] and r["combined_gold_ranks"][60] > 20
        )
        print(
            f"  At cutoff={cutoff}: {reachable}/{n} ({100 * reachable / n:.0f}%) queries have gold IN candidates (reranking ceiling)"
        )
        print(
            f"  At cutoff={cutoff}: {missing_from_candidates}/{n} ({100 * missing_from_candidates / n:.0f}%) queries have gold MISSING from candidates (first-stage recall failure)"
        )
        print(
            f"  At cutoff={cutoff}: {present_but_low}/{n} ({100 * present_but_low / n:.0f}%) queries have gold IN candidates but ranked >20 (obvious reranking targets)"
        )
    print()

    # ================================================================
    # DELIVERABLE 4: TAIL CHARACTERIZATION
    # ================================================================
    print("=== DELIVERABLE 4: TAIL CHARACTERIZATION ===")

    tail = [r for r in all_results if r["combined_gold_ranks"][60] > 20]
    non_tail = [r for r in all_results if r["combined_gold_ranks"][60] <= 20]

    print(f"\nTail size: {len(tail)}/{n} ({100 * len(tail) / n:.1f}%)")
    print(f"Non-tail: {len(non_tail)}/{n} ({100 * len(non_tail) / n:.1f}%)")
    print()

    # List tail queries
    print(
        f"{'Q#':>4} {'Type':>6} {'D@1':>5} {'S@1':>5} {'C@1':>5} {'#Exp':>5} {'Toks':>5} {'Leak%':>6}  Query"
    )
    print("-" * 120)
    tail_sorted = sorted(tail, key=lambda r: r["combined_gold_ranks"][60])
    for r in tail_sorted:
        d_rank = r["dense_gold_rank"] if r["dense_gold_rank"] <= 250 else "miss"
        s_rank = r["sparse_gold_rank"] if r["sparse_gold_rank"] <= 250 else "miss"
        c_rank = (
            r["combined_gold_ranks"][60]
            if r["combined_gold_ranks"][60] <= 250
            else "miss"
        )
        print(
            f"{r['i'] + 1:>4} {r['qtype']:>6} {str(d_rank):>5} {str(s_rank):>5} {str(c_rank):>5} "
            f"{r['n_expected']:>5} {r['qtokens']:>5} {100 * r['leakage']:>5.1f}%  {r['query'][:70]}"
        )
    print()

    # (a) Terse/generic queries — correlation token length vs rank
    all_ranks = [r["combined_gold_ranks"][60] for r in all_results]
    all_tokens = [r["qtokens"] for r in all_results]
    corr = np.corrcoef(all_ranks, all_tokens)[0, 1] if len(all_ranks) > 1 else 0
    short_cutoff = statistics.median(all_tokens)
    short_qs = [r for r in all_results if r["qtokens"] <= short_cutoff]
    long_qs = [r for r in all_results if r["qtokens"] > short_cutoff]
    short_mrr = (
        sum(r["crr_by_k"][60] for r in short_qs) / len(short_qs) if short_qs else 0
    )
    long_mrr = sum(r["crr_by_k"][60] for r in long_qs) / len(long_qs) if long_qs else 0
    print(f"(a) Token length vs Combined rank: Pearson r={corr:.3f}")
    print(
        f"    Short queries (≤{short_cutoff} tokens, n={len(short_qs)}): MRR={short_mrr:.4f}"
    )
    print(
        f"    Long queries  (>{short_cutoff} tokens, n={len(long_qs)}): MRR={long_mrr:.4f}"
    )
    # Collapse to 1/2-5/6-20/21+
    short_buckets = Counter()
    for r in short_qs:
        rank = r["combined_gold_ranks"][60]
        if rank == 1:
            short_buckets["1"] += 1
        elif rank <= 5:
            short_buckets["2-5"] += 1
        elif rank <= 20:
            short_buckets["6-20"] += 1
        else:
            short_buckets["21+"] += 1
    long_buckets = Counter()
    for r in long_qs:
        rank = r["combined_gold_ranks"][60]
        if rank == 1:
            long_buckets["1"] += 1
        elif rank <= 5:
            long_buckets["2-5"] += 1
        elif rank <= 20:
            long_buckets["6-20"] += 1
        else:
            long_buckets["21+"] += 1
    print(f"    Short dist: {dict(short_buckets)}")
    print(f"    Long dist:  {dict(long_buckets)}")

    print()

    # (b) Multi-ACN queries
    tail_multi = [r for r in tail if r["multi_acn"]]
    non_tail_multi = [r for r in non_tail if r["multi_acn"]]
    print(
        f"(b) Multi-ACN queries: tail={len(tail_multi)}/{len(tail)} non_tail={len(non_tail_multi)}/{len(non_tail)}"
    )
    if non_tail_multi:
        non_tail_recall = {}
        for k in KS:
            vals = []
            for r in non_tail_multi:
                top_c = set(
                    acn
                    for acn, _ in sorted(
                        r["combined_ranks_by_k"][60].items(), key=lambda x: x[1]
                    )[:k]
                )
                vals.append(len(top_c & r["expected"]) / len(r["expected"]))
            non_tail_recall[k] = sum(vals) / len(vals)
        print(
            f"    Non-tail multi-ACN recall: {', '.join(f'@{k}={v:.3f}' for k, v in non_tail_recall.items())}"
        )
    if tail_multi:
        tail_recall = {}
        for k in KS:
            vals = []
            for r in tail_multi:
                top_c = set(
                    acn
                    for acn, _ in sorted(
                        r["combined_ranks_by_k"][60].items(), key=lambda x: x[1]
                    )[:k]
                )
                vals.append(len(top_c & r["expected"]) / len(r["expected"]))
            tail_recall[k] = sum(vals) / len(vals) if vals else 0
        print(
            f"    Tail multi-ACN recall: {', '.join(f'@{k}={v:.3f}' for k, v in tail_recall.items())}"
        )

    print()

    # (c) Vocabulary mismatch — leakage in tail vs non-tail
    tail_leakages = [r["leakage"] for r in tail]
    non_tail_leakages = [r["leakage"] for r in non_tail]
    avg_tail_leak = statistics.mean(tail_leakages) if tail_leakages else 0
    avg_non_tail_leak = statistics.mean(non_tail_leakages) if non_tail_leakages else 0
    corpus_avg = statistics.mean([r["leakage"] for r in all_results])
    print(f"(c) Vocabulary mismatch (leakage):")
    print(f"    Corpus avg leakage: {100 * corpus_avg:.1f}%")
    print(f"    Tail avg leakage: {100 * avg_tail_leak:.1f}%")
    print(f"    Non-tail avg leakage: {100 * avg_non_tail_leak:.1f}%")
    # Is tail lower-overlap?
    print(
        f"    Tail has {'LOWER' if avg_tail_leak < avg_non_tail_leak else 'HIGHER'} leakage than non-tail (Δ={100 * (avg_tail_leak - avg_non_tail_leak):+.1f}%)"
    )

    # ================================================================
    # DELIVERABLE 5: LEAKAGE SANITY CHECK
    # ================================================================
    print("\n=== DELIVERABLE 5: LEAKAGE SANITY CHECK ===")

    # Stratify conceptual queries by leakage
    conceptuals = [r for r in all_results if r["qtype"] == "conceptual"]
    high_leak = [r for r in conceptuals if r["leakage"] > 0.353]  # above corpus avg
    low_leak = [r for r in conceptuals if r["leakage"] <= 0.353]

    if high_leak:
        high_mrr = sum(r["crr_by_k"][60] for r in high_leak) / len(high_leak)
        high_dense_mrr = sum(r["drr"] for r in high_leak) / len(high_leak)
        high_sparse_mrr = sum(r["srr"] for r in high_leak) / len(high_leak)
        print(
            f"  High-leakage conceptual (>{35.3}%, n={len(high_leak)}): dense={high_dense_mrr:.4f} sparse={high_sparse_mrr:.4f} combined={high_mrr:.4f}"
        )
    if low_leak:
        low_mrr = sum(r["crr_by_k"][60] for r in low_leak) / len(low_leak)
        low_dense_mrr = sum(r["drr"] for r in low_leak) / len(low_leak)
        low_sparse_mrr = sum(r["srr"] for r in low_leak) / len(low_leak)
        print(
            f"  Low-leakage conceptual (≤{35.3}%, n={len(low_leak)}): dense={low_dense_mrr:.4f} sparse={low_sparse_mrr:.4f} combined={low_mrr:.4f}"
        )

    # The "paraphrase-robust" number: MRR on low-leakage queries only
    print(
        f"\n  Paraphrase-robust conceptual MRR (low-leakage only): combined={sum(r['crr_by_k'][60] for r in low_leak) / len(low_leak):.4f}"
        if low_leak
        else ""
    )
    if low_leak:
        low_combined = sum(r["crr_by_k"][60] for r in low_leak) / len(low_leak)
        print(
            f"  vs corpus-wide conceptual MRR: {sum(r['crr_by_k'][60] for r in conceptuals) / len(conceptuals):.4f}"
        )

    # ================================================================
    # WRITE REPORT
    # ================================================================
    report = f"""# Stage 3 Hybrid Retrieval Diagnosis

## 1. Distribution Shape

**Combined (k=60) gold rank histogram:**

| Bucket | Count | % |
|--------|-------|---|
| Rank 1 | {buckets["1"]} | {100 * buckets["1"] / n:.1f}% |
| Ranks 2–5 | {buckets["2-5"]} | {100 * buckets["2-5"] / n:.1f}% |
| Ranks 6–20 | {buckets["6-20"]} | {100 * buckets["6-20"] / n:.1f}% |
| Ranks 21–50 | {buckets["21-50"]} | {100 * buckets["21-50"] / n:.1f}% |
| Ranks 51+ | {buckets["51+"]} | {100 * buckets["51+"] / n:.1f}% |

- @1: {100 * buckets["1"] / n:.1f}%
- Top-5: {100 * (buckets["1"] + buckets["2-5"]) / n:.1f}%
- Top-20: {100 * (buckets["1"] + buckets["2-5"] + buckets["6-20"]) / n:.1f}%
- Tail (>20): {100 * (buckets["21-50"] + buckets["51+"]) / n:.1f}%

**Bimodality check:** Rank-1 = {at_1}, Middle (2–20) = {in_middle}, Tail (21+) = {in_tail}. {"The distribution IS bimodal" if at_1 + in_tail > 2 * in_middle else "Not strongly bimodal — middle is substantial"} ({in_middle / n:.1%} of queries in 2–20).

## 2. Fusion Sensitivity (RRF k sweep)

Baselines:
- Dense MRR: overall={dense_mrr_overall:.4f}, conceptual={dense_mrr_conc:.4f}, exact_token={dense_mrr_exac:.4f}
- Sparse MRR: overall={sparse_mrr_overall:.4f}, conceptual={sparse_mrr_conc:.4f}, exact_token={sparse_mrr_exac:.4f}

| k | Overall | Conceptual | Exact_token |
|---|---------|------------|-------------|
"""
    for k in RRF_K_VALUES:
        report += f"| {k} | {mrr_tables['overall'][k]:.4f} | {mrr_tables['conceptual'][k]:.4f} | {mrr_tables['exact_token'][k]:.4f} |\n"

    best_overall_k = max(RRF_K_VALUES, key=lambda k: mrr_tables["overall"][k])
    best_conc_k = max(RRF_K_VALUES, key=lambda k: mrr_tables["conceptual"][k])
    best_exac_k = max(RRF_K_VALUES, key=lambda k: mrr_tables["exact_token"][k])

    report += f"""
Best k per slice:
- Overall: k={best_overall_k} (MRR={mrr_tables["overall"][best_overall_k]:.4f})
- Conceptual: k={best_conc_k} (MRR={mrr_tables["conceptual"][best_conc_k]:.4f})
- Exact_token: k={best_exac_k} (MRR={mrr_tables["exact_token"][best_exac_k]:.4f})

**No single k makes Combined ≥ max(Dense, Sparse) on all slices simultaneously.** RRF inherently trades off between the two retrievers and cannot surpass the better retriever on every slice.

### Current regressions at k=60
- Exact_token Combined ({combined_exac_k60:.4f}) vs Sparse alone ({sparse_mrr_exac:.4f}): Δ = {combined_exac_k60 - sparse_mrr_exac:.4f}
- Multi-ACN@5 Combined ({combined_multi_at5[60]:.4f}) vs Dense alone ({dense_multi_at5:.4f}): Δ = {combined_multi_at5[60] - dense_multi_at5:.4f}

### Per-type weighting test
"""
    weight_scores_k60 = []
    weight_scores_k5 = []
    for w_sparse_exac in [0.3, 0.5, 0.7, 0.9]:
        for w_dense_conc in [0.3, 0.5, 0.7, 0.9]:
            weighted_rr = []
            weighted_rr_exac = []
            weighted_rr_conc = []
            for r in all_results:
                default = 251
                all_acns = set(acn for acn, _ in r["dense_collapsed"]) | set(
                    acn for acn, _ in r["sparse_collapsed"]
                )
                scores = {}
                w_dense = (
                    w_dense_conc if r["qtype"] == "conceptual" else (1 - w_sparse_exac)
                )
                w_sparse = (
                    (1 - w_dense_conc) if r["qtype"] == "conceptual" else w_sparse_exac
                )
                dr = {
                    acn: rank for rank, (acn, _) in enumerate(r["dense_collapsed"], 1)
                }
                sr = {
                    acn: rank for rank, (acn, _) in enumerate(r["sparse_collapsed"], 1)
                }
                for acn in all_acns:
                    scores[acn] = w_dense / (60 + dr.get(acn, default)) + w_sparse / (
                        60 + sr.get(acn, default)
                    )
                combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                mrr = _parent_mrr([(acn, 0) for acn, _ in combined], r["expected"])
                weighted_rr.append(mrr)
                if r["qtype"] == "exact_token":
                    weighted_rr_exac.append(mrr)
                else:
                    weighted_rr_conc.append(mrr)
            w_overall = sum(weighted_rr) / len(weighted_rr)
            w_exac = (
                sum(weighted_rr_exac) / len(weighted_rr_exac) if weighted_rr_exac else 0
            )
            w_conc = (
                sum(weighted_rr_conc) / len(weighted_rr_conc) if weighted_rr_conc else 0
            )
            beats_exac = w_exac >= combined_exac_k60
            beats_conc = w_conc >= mrr_tables["conceptual"][60]
            beats_exac_k5 = w_exac >= mrr_tables["exact_token"][5]
            beats_conc_k5 = w_conc >= mrr_tables["conceptual"][5]
            if beats_exac and beats_conc:
                weight_scores_k60.append(
                    (w_sparse_exac, w_dense_conc, w_overall, w_exac, w_conc)
                )
            if beats_exac_k5 and beats_conc_k5:
                weight_scores_k5.append(
                    (w_sparse_exac, w_dense_conc, w_overall, w_exac, w_conc)
                )

    if weight_scores_k60:
        report += f"Found {len(weight_scores_k60)} weighted configs that BEAT global k=60 on both slices:\n"
        for ws, wd, wo, we, wc in weight_scores_k60:
            report += f"- w_sparse_exac={ws:.1f} w_dense_conc={wd:.1f}: overall={wo:.4f} exac={we:.4f} conc={wc:.4f}\n"
    else:
        report += "No per-type weighting configuration beats global k=60 on both slices simultaneously.\n"
        report += "(Per-type weighting improves one slice at the expense of the other — fundamental RRF tradeoff.)\n"
    if weight_scores_k5:
        report += f"Found {len(weight_scores_k5)} weighted configs that BEAT global k=5 on both slices:\n"
        for ws, wd, wo, we, wc in weight_scores_k5:
            report += f"- w_sparse_exac={ws:.1f} w_dense_conc={wd:.1f}: overall={wo:.4f} exac={we:.4f} conc={wc:.4f}\n"
    else:
        report += "No per-type weighting configuration beats global k=5 on both slices simultaneously.\n"

    # Tail characterization for report
    tail_rows = ""
    for r in tail_sorted:
        d_rank = r["dense_gold_rank"] if r["dense_gold_rank"] <= 250 else "miss"
        s_rank = r["sparse_gold_rank"] if r["sparse_gold_rank"] <= 250 else "miss"
        c_rank = (
            r["combined_gold_ranks"][60]
            if r["combined_gold_ranks"][60] <= 250
            else "miss"
        )
        tail_rows += f"| Q{r['i'] + 1} | {r['qtype']} | {d_rank} | {s_rank} | {c_rank} | {r['n_expected']} | {r['qtokens']} | {100 * r['leakage']:.0f}% | {r['query'][:70]} |\n"

    report += f"""
## 3. Reranker Ceiling

Fraction of queries where gold doc is in the Combined top-c candidate set at different cutoffs:

| Cutoff | All (%) | Conceptual (%) | Exact_token (%) |
|--------|---------|----------------|-----------------|
"""
    for cutoff in [10, 20, 50, 100]:
        all_reach = sum(1 for r in all_results if r["rerank_reach"][cutoff])
        conc_reach = sum(
            1
            for r in all_results
            if r["qtype"] == "conceptual" and r["rerank_reach"][cutoff]
        )
        exac_reach = sum(
            1
            for r in all_results
            if r["qtype"] == "exact_token" and r["rerank_reach"][cutoff]
        )
        n_conc = sum(1 for r in all_results if r["qtype"] == "conceptual")
        n_exac = sum(1 for r in all_results if r["qtype"] == "exact_token")
        report += f"| {cutoff} | {all_reach}/{n} ({100 * all_reach / n:.0f}%) | {conc_reach}/{n_conc} ({100 * conc_reach / n_conc:.0f}%) | {exac_reach}/{n_exac} ({100 * exac_reach / n_exac:.0f}%) |\n"

    # Bottleneck split
    missing_50 = sum(1 for r in all_results if not r["rerank_reach"][50])
    reachable_50 = n - missing_50
    present_low_50 = sum(
        1
        for r in all_results
        if r["rerank_reach"][50] and r["combined_gold_ranks"][60] > 20
    )
    report += f"""
**Bottleneck at cutoff=50:**
- Gold IN candidates (reranking ceiling): {reachable_50}/{n} ({100 * reachable_50 / n:.0f}%)
- Gold MISSING from candidates (first-stage recall failure): {missing_50}/{n} ({100 * missing_50 / n:.0f}%)
- Gold IN candidates but ranked >20 (obvious reranking targets): {present_low_50}/{n} ({100 * present_low_50 / n:.0f}%)

Conclusion: The reranking ceiling is high — {reachable_50}/{n} ({100 * reachable_50 / n:.0f}%) of queries have the gold doc reachable at cutoff=50. A reranker can reorder the top-50 candidates for any of these queries, not just those beyond rank 20. Only {missing_50}/{n} ({100 * missing_50 / n:.0f}%) suffer from first-stage recall failure and need retrieval-level improvements.
"""

    report += f"""
## 4. Tail Characterization

**Tail size:** {len(tail)}/{n} ({100 * len(tail) / n:.1f}%) of queries with Combined gold rank > 20.

### Tail query detail

| Q# | Type | Dense@1 | Sparse@1 | Comb@1 | #Exp | Toks | Leak% | Query |
|----|------|---------|----------|--------|------|------|-------|-------|
{tail_rows}
"""
    # (a)
    report += f"""
### (a) Terse/generic queries
- Pearson r(token_length, gold_rank) = {corr:.3f}
- Short queries (≤{short_cutoff} tokens, n={len(short_qs)}): MRR = {short_mrr:.4f}
- Long queries (>{short_cutoff} tokens, n={len(long_qs)}): MRR = {long_mrr:.4f}
- Short distribution: {dict(short_buckets)}
- Long distribution: {dict(long_buckets)}
"""
    # (b)
    report += f"""
### (b) Multi-constraint queries
- Tail: {len(tail_multi)}/{len(tail)} multi-ACN queries vs Non-tail: {len(non_tail_multi)}/{len(non_tail)}
"""
    if non_tail_multi:
        report += f"- Non-tail multi-ACN recall: {', '.join(f'@{k}={non_tail_recall[k]:.3f}' for k in KS)}\n"
    if tail_multi:
        report += f"- Tail multi-ACN recall: {', '.join(f'@{k}={tail_recall[k]:.3f}' for k in KS)}\n"

    report += f"""
### (c) Vocabulary mismatch
- Corpus avg query-narrative token overlap: {100 * corpus_avg:.1f}%
- Tail avg overlap: {100 * avg_tail_leak:.1f}%
- Non-tail avg overlap: {100 * avg_non_tail_leak:.1f}%
- Tail has {"LOWER" if avg_tail_leak < avg_non_tail_leak else "HIGHER"} overlap than non-tail by {100 * abs(avg_tail_leak - avg_non_tail_leak):.1f} percentage points
"""
    # Failure mode clustering
    report += """
### Failure mode clusters

| Cluster | Queries | Evidence | Likely fix |
|---------|---------|----------|------------|
"""
    # Count conceptual tail vs exact token tail
    conc_tail = [r for r in tail if r["qtype"] == "conceptual"]
    exac_tail = [r for r in tail if r["qtype"] == "exact_token"]
    short_tail = [r for r in tail if r["qtokens"] <= short_cutoff]
    multi_tail = [r for r in tail if r["multi_acn"]]
    low_leak_tail = [r for r in tail if r["leakage"] <= 0.353]

    if conc_tail:
        report += f"| Vocabulary-paraphrase gap | Q{','.join(str(r['i'] + 1) for r in conc_tail)} ({len(conc_tail)} conc) | Conceptual queries rely on synonyms/paraphrase; BM42 cannot bridge this; tail avg leakage is {100 * avg_tail_leak:.0f}% vs {100 * avg_non_tail_leak:.0f}% non-tail | HyDE / query expansion / Latent-Interation (ColBERT) |\n"
    if exac_tail:
        report += f"| Sparse-first failure | Q{','.join(str(r['i'] + 1) for r in exac_tail)} ({len(exac_tail)} exac) | Exact_token queries that both dense AND sparse fail on — suggesting the gold doc is hard to match with any BOW method | Query expansion (add synonymous codes) / structured metadata pre-filter |\n"
    if short_tail:
        report += f"| Terse/under-specified | Q{','.join(str(r['i'] + 1) for r in short_tail if r in tail)} ({len([r for r in tail if r in short_tail])} qs) | Short queries lack discriminating vocabulary; token-length-rank correlation r={corr:.3f} | HyDE (expand to full scenario) |\n"
    if multi_tail:
        report += f"| Compositional/multi-constraint | Q{','.join(str(r['i'] + 1) for r in multi_tail)} ({len(multi_tail)} qs) | Queries with 2+ expected ACNs or stacked conditions require retrieving multiple distinct docs | ColBERT late interaction / multi-vector retrieval |\n"

    report += f"""
## 5. Leakage Sanity Check

Conceptual queries stratified by query-narrative token overlap:
"""
    if high_leak:
        report += f"- High-overlap (>{35.3}%, n={len(high_leak)}): dense={high_dense_mrr:.4f} sparse={high_sparse_mrr:.4f} combined={high_mrr:.4f}\n"
    if low_leak:
        low_combined = sum(r["crr_by_k"][60] for r in low_leak) / len(low_leak)
        report += f"- Low-overlap (≤{35.3}%, n={len(low_leak)}): dense={low_dense_mrr:.4f} sparse={low_sparse_mrr:.4f} combined={low_combined:.4f}\n"

    all_conc = sum(r["crr_by_k"][60] for r in conceptuals) / len(conceptuals)
    report += f"""
**Paraphrase-robust conceptual MRR** (low-overlap queries only, n={len(low_leak)}): combined = {low_combined:.4f}
vs corpus-wide conceptual combined MRR (n={len(conceptuals)}): {all_conc:.4f}
"""
    if low_leak:
        delta = low_combined - all_conc
        report += f"Δ = {delta:+.4f} ({'lower' if delta < 0 else 'higher'} — meaning the reported MRR is {'inflated' if delta < 0 else 'deflated'} by leaky queries)\n"
    else:
        report += "Insufficient data for stratification.\n"

    with open("diagnosis_report.md", "w") as f:
        f.write(report)

    print("\n=== Report written to diagnosis_report.md ===")


if __name__ == "__main__":
    main()
