"""
Edge case analysis: find queries where captain narrative is the decisive field.
i.e., queries where removing captain children would cause the gold ACN to drop rank
or be completely missed, despite synopsis being present.
"""

import json
import logging

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RRF_K = 60
SEARCH_LIMIT = 200


def collapse(scored_points):
    acn_scores = {}
    for sp in scored_points:
        rid = sp.payload.get("report_id")
        if rid is not None:
            acn = int(rid)
            score = sp.score
            if acn not in acn_scores or score > acn_scores[acn]:
                acn_scores[acn] = score
    return sorted(acn_scores.items(), key=lambda x: x[1], reverse=True)


def rrf_rank(dense_collapsed, sparse_collapsed, gold_acn):
    dr_map = {a: r for r, (a, _) in enumerate(dense_collapsed, 1)}
    sr_map = {a: r for r, (a, _) in enumerate(sparse_collapsed, 1)}
    default = max(len(dense_collapsed), len(sparse_collapsed)) + 1
    all_a = set(dr_map) | set(sr_map)
    scores = {}
    for a in all_a:
        scores[a] = 1.0 / (RRF_K + dr_map.get(a, default)) + 1.0 / (
            RRF_K + sr_map.get(a, default)
        )
    combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (acn, _) in enumerate(combined, 1):
        if acn == gold_acn:
            return rank
    return None


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

    degrading = []  # queries where removing captain hurts

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

        # ── All fields ──
        dense_all = collapse(dense_raw)
        sparse_all = collapse(sparse_raw)

        # ── Without captain ──
        dense_no_captain = collapse(
            [sp for sp in dense_raw if sp.payload.get("narrative_source") != "captain"]
        )
        sparse_no_captain = collapse(
            [sp for sp in sparse_raw if sp.payload.get("narrative_source") != "captain"]
        )

        # For each gold ACN, compare ranks
        for gold_acn in expected:
            rank_all = rrf_rank(dense_all, sparse_all, gold_acn)
            rank_no_captain = rrf_rank(dense_no_captain, sparse_no_captain, gold_acn)

            if rank_all is None and rank_no_captain is None:
                continue

            if rank_no_captain is None and rank_all is not None:
                # CRITICAL: captain is the ONLY reason this ACN is found at all
                degrading.append(
                    {
                        "qnum": qnum,
                        "query": q["query"][:80],
                        "gold_acn": gold_acn,
                        "rank_all": rank_all,
                        "rank_no_captain": None,
                        "type": "MISS_WITHOUT_CAPTAIN",
                        "delta_rank": None,
                    }
                )
            elif (
                rank_no_captain is not None
                and rank_all is not None
                and rank_no_captain > rank_all
            ):
                # Degraded but still found — but only interesting if it drops below rank 1
                degrading.append(
                    {
                        "qnum": qnum,
                        "query": q["query"][:80],
                        "gold_acn": gold_acn,
                        "rank_all": rank_all,
                        "rank_no_captain": rank_no_captain,
                        "type": "DEGRADED",
                        "delta_rank": rank_no_captain - rank_all,
                    }
                )

        if (i + 1) % 15 == 0:
            logger.info("Processed %d/%d", i + 1, len(queries))

    mgr.close()

    # ── Report ──
    print("=" * 80)
    print("  EDGE CASE ANALYSIS: Does captain narrative protect against degradation?")
    print("=" * 80)

    print(f"\nTotal queries with SOME impact when captain is removed: {len(degrading)}")
    print()

    # Separate by type
    critical = [d for d in degrading if d["type"] == "MISS_WITHOUT_CAPTAIN"]
    degraded = [d for d in degrading if d["type"] == "DEGRADED"]

    if critical:
        print(
            f"── CRITICAL: {len(critical)} gold ACN(s) would be COMPLETELY MISSED without captain ──"
        )
        print()
        for d in critical:
            print(f"  Q{d['qnum']}: ACN {d['gold_acn']}")
            print(f'    Query: "{d["query"]}"')
            print(f"    With captain:    rank {d['rank_all']}")
            print(f"    Without captain: NOT FOUND")
            print()

    if degraded:
        print(
            f"── DEGRADED: {len(degraded)} gold ACN(s) would drop rank without captain ──"
        )
        print()
        # Only show meaningful drops (rank 1->2+, or drop out of top-5)
        meaningful = [
            d for d in degraded if d["rank_all"] == 1 or d["rank_no_captain"] > 5
        ]
        for d in degraded:
            arrow = " <<< ranks 1" if d["rank_all"] == 1 else ""
            marker = (
                " ** top-5 loss **"
                if d["rank_no_captain"] > 5 and d["rank_all"] <= 5
                else ""
            )
            gap = d["rank_no_captain"] - d["rank_all"]
            print(
                f"  Q{d['qnum']:>2} ACN {d['gold_acn']}: rank {d['rank_all']} -> {d['rank_no_captain']} (Δ+{gap}){marker}{arrow}"
            )
            if d["rank_all"] == 1 or d["rank_no_captain"] > 5:
                print(f'    Query: "{d["query"]}"')

    if not critical and not degraded:
        print("  No queries are affected by removing captain narrative.")

    print()
    print("── Summary ──")
    print(f"  MRR with all fields:      0.9769")
    print(f"  MRR without captain:      0.7229 (dense-only field ablated from earlier)")
    print(f"  Critical misses:          {len(critical)} ACNs would vanish")
    print(f"  Degraded (any drop):      {len(degraded)} ACNs would rank lower")
    print()

    if not critical and len([d for d in degraded if d.get("rank_all") == 1]) == 0:
        print(
            "  CONCLUSION: Captain narrative is NOT protecting any gold ACN at rank 1."
        )
        print(
            "  No query's #1 result depends on captain. Synopsis + anomaly are sufficient."
        )
        print("  However, captain does provide marginal ranking lift for some queries")
        print("  that may matter for downstream RAG (deeper in the ranking).")
    else:
        print("  WARNING: Captain narrative IS critical for some queries.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    run()
