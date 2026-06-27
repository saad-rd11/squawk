"""
Experiment: augment synopsis child text with structured fields (anomaly, flight phase)
to bridge vocabulary gap. Tests: does enriched text improve reranking?

Approach: for each query, take top-30 RRF results, build augmented text for each
candidate (synopsis + anomaly + flight phase), score with cross-encoder, rerank.
"""

import json
import logging
import time

from qdrant_client.models import SparseVector
from sentence_transformers import CrossEncoder

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RRF_K = 60
SEARCH_LIMIT = 200
RERANK_DEPTH = 30


def collapse(raw):
    out = {}
    for sp in raw:
        rid = sp.payload.get("report_id")
        if rid is not None:
            a = int(rid)
            if a not in out or sp.score > out[a]["score"]:
                out[a] = {"score": sp.score}
    return sorted(out.items(), key=lambda x: x[1]["score"], reverse=True)


def build_augmented(row):
    """Build enriched text: structured fields + synopsis."""
    synopsis = str(row.get("Synopsis", ""))
    anomaly = str(row.get("Anomaly", ""))
    flight_phase = str(row.get("Flight Phase", ""))
    primary_problem = str(row.get("Primary Problem", ""))

    parts = []
    if flight_phase and flight_phase.lower() != "nan":
        parts.append(f"[Flight Phase: {flight_phase}]")
    if anomaly and anomaly.lower() != "nan":
        parts.append(f"[Anomaly: {anomaly}]")
    if primary_problem and primary_problem.lower() != "nan":
        parts.append(f"[Problem: {primary_problem}]")
    parts.append(synopsis)
    return " ".join(parts)


def main():
    with open("eval_queries.json") as f:
        queries = json.load(f)

    import pandas as pd

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)
    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))
    pool_df = df[df["ACN"].isin(pool_acns)].set_index("ACN")

    config = DEFAULT_CONFIG

    ded = DenseEmbedder(
        model_name=config.dense_model, batch_size=config.embed_batch_size
    )
    sed = SparseEmbedder(
        model_name=config.sparse_model, batch_size=config.embed_batch_size
    )
    mgr = Stage3Collection(path="./qdrant_storage")

    qtexts = [q["query"] for q in queries]
    logger.info("Embedding queries...")
    dvecs = ded.embed(qtexts)
    svecs = sed.embed(qtexts)

    logger.info("Loading cross-encoder...")
    t0 = time.time()
    cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info(f"Loaded in {time.time() - t0:.1f}s")

    baseline_rr = []
    plain_rr = []
    augmented_rr = []
    per_query = []

    for i, q in enumerate(queries):
        expected = set(q["expected_acns"])
        qnum = i + 1
        qtext = q["query"]
        qtype = q.get("type", "conceptual")

        # Retrieve
        dense_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dvecs[i],
            using="dense",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points
        sparse_raw = mgr.client.query_points(
            collection_name=COLLECTION_NAME,
            query=SparseVector(
                indices=list(svecs[i].keys()), values=list(svecs[i].values())
            ),
            using="sparse",
            limit=SEARCH_LIMIT,
            with_payload=["report_id"],
        ).points

        dc = collapse(dense_raw)
        sc = collapse(sparse_raw)

        dm = {a: r for r, (a, _) in enumerate(dc, 1)}
        sm = {a: r for r, (a, _) in enumerate(sc, 1)}
        default = max(len(dc), len(sc)) + 1
        all_a = set(dm) | set(sm)
        scores = {}
        for a in all_a:
            scores[a] = 1.0 / (RRF_K + dm.get(a, default)) + 1.0 / (
                RRF_K + sm.get(a, default)
            )
        rrf_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Baseline MRR
        baseline_mrr = 0.0
        for rank, (acn, _) in enumerate(rrf_results, 1):
            if acn in expected:
                baseline_mrr = 1.0 / rank
                break
        baseline_rr.append(baseline_mrr)

        top_k = rrf_results[:RERANK_DEPTH]

        # Build pairs for BOTH plain and augmented
        plain_pairs = []
        aug_pairs = []
        candidate_acns = []
        for acn, _ in top_k:
            try:
                row = pool_df.loc[acn]
            except KeyError:
                continue
            syn = str(row.get("Synopsis", ""))
            aug_text = build_augmented(row)
            plain_pairs.append((qtext, syn))
            aug_pairs.append((qtext, aug_text))
            candidate_acns.append(acn)

        # Score both
        if plain_pairs:
            plain_scores = cross_enc.predict(plain_pairs, show_progress_bar=False)
            aug_scores = cross_enc.predict(aug_pairs, show_progress_bar=False)
        else:
            plain_scores = []
            aug_scores = []

        plain_reranked = sorted(
            zip(candidate_acns, plain_scores), key=lambda x: x[1], reverse=True
        )
        aug_reranked = sorted(
            zip(candidate_acns, aug_scores), key=lambda x: x[1], reverse=True
        )

        # MRR for both
        plain_mrr = 0.0
        for rank, (acn, _) in enumerate(plain_reranked, 1):
            if acn in expected:
                plain_mrr = 1.0 / rank
                break
        plain_rr.append(plain_mrr)

        aug_mrr = 0.0
        for rank, (acn, _) in enumerate(aug_reranked, 1):
            if acn in expected:
                aug_mrr = 1.0 / rank
                break
        augmented_rr.append(aug_mrr)

        per_query.append(
            {
                "qnum": qnum,
                "query": qtext[:60],
                "qtype": qtype,
                "baseline_mrr": baseline_mrr,
                "plain_ce_mrr": plain_mrr,
                "augmented_mrr": aug_mrr,
                "aug_improves_plain": aug_mrr > plain_mrr,
                "aug_degrades_plain": aug_mrr < plain_mrr,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(queries)}")

    mgr.close()

    n = len(queries)
    b = sum(baseline_rr) / n
    p = sum(plain_rr) / n
    a = sum(augmented_rr) / n

    # ── Report ──
    print()
    print("=" * 80)
    print("  SYNOPSIS AUGMENTATION EXPERIMENT")
    print("  Hypothesis: Adding [Flight Phase][Anomaly] prefix to synopsis")
    print("  bridges vocabulary gap in cross-encoder scoring")
    print("=" * 80)
    print()
    print(
        f"  {'':>30}  {'Baseline':>10}  {'Plain CE':>10}  {'Augmented':>10}  {'Δ (aug vs plain)':>16}"
    )
    print(f"  {'-' * 80}")
    print(f"  {'Overall MRR':>30}  {b:.4f}  {p:.4f}  {a:.4f}  {a - p:+.4f}")

    for qtype in sorted(set(q["qtype"] for q in per_query)):
        subset = [q for q in per_query if q["qtype"] == qtype]
        sb = sum(q["baseline_mrr"] for q in subset) / len(subset)
        sp = sum(q["plain_ce_mrr"] for q in subset) / len(subset)
        sa = sum(q["augmented_mrr"] for q in subset) / len(subset)
        print(
            f"  {f'[{qtype}] ({len(subset)} queries)':>30}  {sb:.4f}  {sp:.4f}  {sa:.4f}  {sa - sp:+.4f}"
        )

    # Counts
    n_improved = sum(1 for q in per_query if q["aug_improves_plain"])
    n_degraded = sum(1 for q in per_query if q["aug_degrades_plain"])
    n_same = n - n_improved - n_degraded
    print(
        f"  {'Aug improves vs plain CE':>30}  {n_improved:>4}/{n_degraded:>4}/{n_same:>4}"
    )

    # Also count: augmented vs baseline
    n_aug_vs_base = sum(1 for q in per_query if q["augmented_mrr"] > q["baseline_mrr"])
    n_aug_vs_base_down = sum(
        1 for q in per_query if q["augmented_mrr"] < q["baseline_mrr"]
    )
    n_aug_vs_base_same = n - n_aug_vs_base - n_aug_vs_base_down
    print(
        f"  {'Aug vs baseline':>30}  {n_aug_vs_base:>4}/{n_aug_vs_base_down:>4}/{n_aug_vs_base_same:>4}"
    )

    # Per-query
    print()
    print("  ── Per-query (sorted by delta) ──")
    print(
        f"  {'Q':>3} {'Type':>6}  {'Base':>6} {'Plain':>6} {'Aug':>6} {'Δ Aug-Plain':>12}  Query"
    )
    print(f"  {'-' * 85}")
    for q in sorted(per_query, key=lambda x: x["augmented_mrr"] - x["plain_ce_mrr"]):
        delta = q["augmented_mrr"] - q["plain_ce_mrr"]
        marker = " ↑" if delta > 0 else (" ↓" if delta < 0 else "")
        print(
            f"  {q['qnum']:>3} {q['qtype'][:6]:>6}  "
            f"{q['baseline_mrr']:.4f} {q['plain_ce_mrr']:.4f} {q['augmented_mrr']:.4f} "
            f"{delta:+.4f}{marker}  {q['query']}"
        )

    # ── Summary ──
    print()
    print("=" * 80)
    print(f"  Baseline MRR:        {b:.4f}")
    print(f"  Plain CE rerank:     {p:.4f}")
    print(f"  Augmented CE rerank: {a:.4f}")
    print(f"  Augmented vs plain:  {a - p:+.4f}")
    print(f"  Augmented vs base:   {a - b:+.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
