"""
Field contribution analysis: for each query, determine which child type
(narrative_source) drives the gold ACN's rank.
Identifies which fields do the work, which are noise, and what signal each adds.
"""

import json
import logging
from collections import Counter, defaultdict

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 200


def run():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )
    mgr = Stage3Collection(path="./qdrant_storage")

    qtexts = [q["query"] for q in queries]
    dvecs = ded.embed(qtexts)
    svecs = sed.embed(qtexts)

    # Per-field aggregators
    field_stats = defaultdict(
        lambda: {
            "gold_scores": [],
            "non_gold_scores": [],
            "wins": 0,
            "queries_with_win": set(),
            "queries_with_gold": set(),
        }
    )

    # By-field-only MRR and per-query gold winners
    field_only_mrr = defaultdict(list)
    per_query = []

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        qnum = i + 1

        dense_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dvecs[i],
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id", "narrative_source"],
        ).points

        sparse_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(
                indices=list(svecs[i].keys()), values=list(svecs[i].values())
            ),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id", "narrative_source"],
        ).points

        # ── Per-field: best score per gold ACN ──
        field_best_for_gold = defaultdict(lambda: defaultdict(float))
        for sp in dense_raw:
            rid = sp.payload.get("report_id")
            src = sp.payload.get("narrative_source", "unknown")
            if rid is not None:
                acn = int(rid)
                if acn in expected:
                    if sp.score > field_best_for_gold[src].get(acn, -1):
                        field_best_for_gold[src][acn] = sp.score

        # Which field had the max score for each gold ACN?
        gold_acn_best_field = {}
        for acn in expected:
            best_src = None
            best_score = -1
            for src, acn_scores in field_best_for_gold.items():
                sc = acn_scores.get(acn)
                if sc is not None and sc > best_score:
                    best_score = sc
                    best_src = src
            if best_src:
                gold_acn_best_field[acn] = (best_src, best_score)

        # Record field wins and scores
        for src, acn_scores in field_best_for_gold.items():
            for acn, sc in acn_scores.items():
                field_stats[src]["gold_scores"].append(sc)
                field_stats[src]["queries_with_gold"].add(qnum)
                if src == gold_acn_best_field.get(acn, (None, None))[0]:
                    field_stats[src]["wins"] += 1
                    field_stats[src]["queries_with_win"].add(qnum)

        # Non-gold scores per field
        non_gold_sources = defaultdict(list)
        for sp in dense_raw:
            rid = sp.payload.get("report_id")
            src = sp.payload.get("narrative_source", "unknown")
            if rid is not None and int(rid) not in expected:
                non_gold_sources[src].append(sp.score)
        for src, scores in non_gold_sources.items():
            field_stats[src]["non_gold_scores"].extend(scores)

        # ── Per-field MRR (dense-only, per-field clustering) ──
        field_collapsed = defaultdict(dict)
        for sp in dense_raw:
            rid = sp.payload.get("report_id")
            src = sp.payload.get("narrative_source", "unknown")
            if rid is not None:
                acn = int(rid)
                if (
                    acn not in field_collapsed[src]
                    or sp.score > field_collapsed[src][acn]
                ):
                    field_collapsed[src][acn] = sp.score
        for src, acn_scores in field_collapsed.items():
            ranked = sorted(acn_scores.items(), key=lambda x: x[1], reverse=True)
            mrr = 0.0
            for rank, (acn, _) in enumerate(ranked, 1):
                if acn in expected:
                    mrr = 1.0 / rank
                    break
            field_only_mrr[src].append(mrr)

        per_query.append(
            {
                "qnum": qnum,
                "query": q["query"][:60],
                "expected": expected,
                "gold_winners": gold_acn_best_field,
            }
        )

        if (i + 1) % 15 == 0:
            logger.info("Processed %d/%d", i + 1, len(queries))

    mgr.close()

    n = len(queries)

    # ================================================================
    # REPORT
    # ================================================================
    print("=" * 80)
    print("  FIELD CONTRIBUTION ANALYSIS — Which child types drive retrieval?")
    print("=" * 80)

    # ── 1. Win counts ──
    print("\n── 1. Gold ACN 'Win' Counts by Field (dense only) ──")
    print("     For each gold ACN, which child type had the highest dense score?")
    print()
    print(
        f"  {'Field':>20}  {'Wins':>6}  {'% of gold':>9}  "
        f"{'Q with gold':>11}  {'Q with win':>10}  {'Avg gold score':>14}"
    )
    print("  " + "-" * 75)
    total_wins = sum(fs["wins"] for fs in field_stats.values())
    for src in sorted(field_stats, key=lambda s: field_stats[s]["wins"], reverse=True):
        fs = field_stats[src]
        wins = fs["wins"]
        pct = wins / total_wins * 100 if total_wins > 0 else 0
        q_gold = len(fs["queries_with_gold"])
        q_win = len(fs["queries_with_win"])
        avg_gold = (
            sum(fs["gold_scores"]) / len(fs["gold_scores"]) if fs["gold_scores"] else 0
        )
        print(
            f"  {src:>20}  {wins:>6}  {pct:>8.1f}%  {q_gold:>11}  {q_win:>10}  {avg_gold:.4f}"
        )

    # ── 2. Gold vs non-gold separation ──
    print("\n── 2. Signal Separation: Gold vs Non-Gold Avg Score (dense) ──")
    print("     Higher delta = better at finding gold ACNs vs burying them in noise")
    print()
    print(
        f"  {'Field':>20}  {'Avg gold':>9}  {'Avg non-gold':>12}  {'Delta':>7}  {'Discrimination':>15}"
    )
    print("  " + "-" * 65)
    for src in sorted(
        field_stats,
        key=lambda s: (
            sum(field_stats[s]["gold_scores"]) / len(field_stats[s]["gold_scores"])
            if field_stats[s]["gold_scores"]
            else 0
        ),
        reverse=True,
    ):
        fs = field_stats[src]
        avg_g = (
            sum(fs["gold_scores"]) / len(fs["gold_scores"]) if fs["gold_scores"] else 0
        )
        avg_ng = (
            sum(fs["non_gold_scores"]) / len(fs["non_gold_scores"])
            if fs["non_gold_scores"]
            else 0
        )
        delta = avg_g - avg_ng
        if delta > 0.15:
            disc = "Excellent"
        elif delta > 0.08:
            disc = "Good"
        elif delta > 0.03:
            disc = "Weak"
        else:
            disc = "Noise"
        print(f"  {src:>20}  {avg_g:.4f}  {avg_ng:>8.4f}  {delta:+.4f}  {disc:>15}")

    # ── 3. Per-query winning fields ──
    print("\n── 3. Per-Query Winning Fields ──")
    print("     Shows which field scored highest for each gold ACN, per query")
    print()
    for row in per_query:
        winners = row["gold_winners"]
        if not winners:
            continue
        field_counts = Counter(src for src, _ in winners.values())
        win_str = ", ".join(f"{src}={cnt}" for src, cnt in field_counts.most_common())
        expected_str = ", ".join(str(a) for a in sorted(row["expected"]))
        print(f"  Q{row['qnum']:>2}: {win_str:40}  gold={expected_str}")

    # ── 4. Ablation: dense-only MRR per field ──
    print("\n── 4. Dense-Only MRR by Field (simulated per-field collapse) ──")
    print("     What MRR would each field achieve if it were the ONLY child type?")
    print()
    print(f"  {'Field':>20}  {'MRR (dense)':>12}  {'Hit@1':>7}  {'vs overall':>10}")
    print("  " + "-" * 55)
    for src in sorted(
        field_only_mrr,
        key=lambda s: sum(field_only_mrr[s]) / len(field_only_mrr[s]),
        reverse=True,
    ):
        vals = field_only_mrr[src]
        avg_mrr = sum(vals) / n
        hit1 = sum(1 for v in vals if v == 1.0) / n * 100
        print(f"  {src:>20}  {avg_mrr:.4f}           {hit1:>5.1f}%")

    # ── 5. Summary ──
    print("\n── 5. Summary Verdict ──")
    print()
    # Recompute overall dense-only MRR with a fresh query
    mgr2 = Stage3Collection(path="./qdrant_storage")
    overall_dense_mrr = 0.0
    overall_dense_hits = 0
    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        raw = mgr2.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dvecs[i],
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        collapsed = {}
        for sp in raw:
            rid = sp.payload.get("report_id")
            if rid is not None:
                acn = int(rid)
                if acn not in collapsed or sp.score > collapsed[acn]:
                    collapsed[acn] = sp.score
        ranked = sorted(collapsed.items(), key=lambda x: x[1], reverse=True)
        for rank, (acn, _) in enumerate(ranked, 1):
            if acn in expected:
                overall_dense_mrr += 1.0 / rank
                if rank == 1:
                    overall_dense_hits += 1
                break
    overall_dense_mrr /= n
    overall_dense_hit1 = overall_dense_hits / n * 100
    mgr2.close()

    print(
        f"  Overall dense-only MRR (all fields): {overall_dense_mrr:.4f} (Hit@1={overall_dense_hit1:.1f}%)"
    )
    print()

    for src in sorted(field_stats, key=lambda s: field_stats[s]["wins"], reverse=True):
        fs = field_stats[src]
        wins = fs["wins"]
        pct = wins / total_wins * 100 if total_wins > 0 else 0
        avg_g = (
            sum(fs["gold_scores"]) / len(fs["gold_scores"]) if fs["gold_scores"] else 0
        )
        avg_ng = (
            sum(fs["non_gold_scores"]) / len(fs["non_gold_scores"])
            if fs["non_gold_scores"]
            else 0
        )
        delta = avg_g - avg_ng
        if delta > 0.15:
            rating = "Strong signal"
        elif delta > 0.08:
            rating = "Useful"
        elif delta > 0.03:
            rating = "Weak / borderline noise"
        else:
            rating = "Noise (remove?)"

        mrr_vals = field_only_mrr.get(src, [0])
        avg_mrr = sum(mrr_vals) / len(mrr_vals) if mrr_vals else 0
        print(
            f"  {src:>20}: wins={wins}/{total_wins} ({pct:.0f}%), "
            f"delta={delta:+.4f}, "
            f"MRR={avg_mrr:.4f}  ->  {rating}"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    run()
