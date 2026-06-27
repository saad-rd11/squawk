"""
Compare retrieval results before and after anomaly child addition.
Loads baseline_results.json, runs identical eval, produces anomaly_child_findings.md.

Usage:
    ./venv/bin/python -m pipeline.compare_anomaly_child
"""

import json
import logging
import statistics
from collections import defaultdict

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 300
RRF_K = 60
RRF_K_VALUES = [5, 10, 20, 30, 40, 60, 80]
LEAKAGE_THRESHOLD = 0.353


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


def compute_leakage(query_text, narrative_text):
    qtokens = set(query_text.lower().split())
    ntokens = set(narrative_text.lower().split())
    if not qtokens:
        return 0.0
    return len(qtokens & ntokens) / len(qtokens)


def run_eval():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    import pandas as pd

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)

    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))
    pool_df = df[df["ACN"].isin(pool_acns)].set_index("ACN")

    # Load points for point count
    points_count = 0
    with open("points.jsonl") as f:
        for _ in f:
            points_count += 1

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

    all_results = []
    default_rank = SEARCH_LIMIT + 1

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

        dense_ranks = {acn: rank for rank, (acn, _) in enumerate(dense_collapsed, 1)}
        sparse_ranks = {acn: rank for rank, (acn, _) in enumerate(sparse_collapsed, 1)}

        combined_ranks_by_k = {}
        all_acns = set(dense_ranks) | set(sparse_ranks)

        for k in RRF_K_VALUES:
            scores = {}
            for acn in all_acns:
                scores[acn] = 1.0 / (k + dense_ranks.get(acn, default_rank)) + 1.0 / (
                    k + sparse_ranks.get(acn, default_rank)
                )
            combined_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            combined_ranks_by_k[k] = {
                acn: rank for rank, (acn, _) in enumerate(combined_sorted, 1)
            }

        gold_acn = next(iter(expected), None)

        drr = _parent_mrr(dense_collapsed, expected)
        srr = _parent_mrr(sparse_collapsed, expected)
        crr_by_k = {}
        for k in RRF_K_VALUES:
            sorted_c = sorted(combined_ranks_by_k[k].items(), key=lambda x: x[1])
            crr_by_k[k] = _parent_mrr([(acn, 0) for acn, _ in sorted_c], expected)

        narrs = []
        for acn in expected:
            try:
                narrs.append(str(pool_df.loc[acn, "Narrative"]))
            except KeyError:
                pass
        full_narr = " ".join(narrs)
        leakage = compute_leakage(q["query"], full_narr)

        combined_sorted_k60 = sorted(
            combined_ranks_by_k[60].items(), key=lambda x: x[1]
        )
        top10_combined = [
            {"acn": acn, "rank": rank} for acn, rank in combined_sorted_k60[:10]
        ]
        top10_dense = [
            {"acn": acn, "score": score} for acn, score in dense_collapsed[:10]
        ]
        top10_sparse = [
            {"acn": acn, "score": score} for acn, score in sparse_collapsed[:10]
        ]

        all_results.append(
            {
                "i": i,
                "q_num": i + 1,
                "query": q["query"],
                "qtype": q.get("type", "conceptual"),
                "expected": list(expected),
                "dense_gold_rank": dense_ranks.get(gold_acn, default_rank)
                if gold_acn
                else default_rank,
                "sparse_gold_rank": sparse_ranks.get(gold_acn, default_rank)
                if gold_acn
                else default_rank,
                "combined_gold_ranks": {
                    k: ranks.get(gold_acn, default_rank) if gold_acn else default_rank
                    for k, ranks in combined_ranks_by_k.items()
                },
                "drr": drr,
                "srr": srr,
                "crr_by_k": crr_by_k,
                "leakage": leakage,
                "n_expected": len(expected),
                "top10_dense": top10_dense,
                "top10_sparse": top10_sparse,
                "top10_combined": top10_combined,
            }
        )

    mgr.close()
    return all_results, info.points_count, points_count


def get_baseline_mrr(results, k=60):
    vals = [r["crr_by_k"][k] for r in results]
    return statistics.mean(vals) if vals else 0.0


def classify_queries(results):
    CLASSIFICATION_MAP = {
        "ramp worker refusing": "ANSWERABLE",
        "data-entry keyboards": "ANSWERABLE",
        "smoke or a burning smell coming from the control yoke": "ANSWERABLE",
        "cabin fumes making flight attendants physically ill": "ANSWERABLE",
        "widebody crew getting a cargo door warning": "ANSWERABLE",
        "technician inadvertently discharging an engine fire": "ANSWERABLE",
        "helicopter that had to autorotate": "ANSWERABLE",
        "drone exceeding its authorized altitude": "ANSWERABLE",
        "controllers losing communication with a large drone": "BURIED",
        "air ambulance helicopter having a near miss": "ANSWERABLE",
        "a320 losing braking on an icy ramp": "ANSWERABLE",
        "pilot landing without a clearance": "BURIED",
        "baggage cart striking a parked aircraft": "ANSWERABLE",
        "gear-up landing": "BURIED",
        "air carrier altitude or speed deviation where the captain blamed reduced": "BURIED",
    }
    out = {}
    for r in results:
        qtext = r["query"].lower()
        cls = "UNCLASSIFIED"
        for prefix, label in CLASSIFICATION_MAP.items():
            if prefix.lower() in qtext:
                cls = label
                break
        out[r["q_num"]] = cls
    return out


def main():
    # Load baseline
    with open("baseline_results.json") as f:
        baseline = json.load(f)
    baseline_results = baseline["results"]
    baseline_point_count = baseline["point_count"]
    # JSON serializes int keys as strings — convert back
    for r in baseline_results:
        r["combined_gold_ranks"] = {
            int(k): v for k, v in r["combined_gold_ranks"].items()
        }
        r["crr_by_k"] = {int(k): v for k, v in r["crr_by_k"].items()}
    logger.info("Loaded baseline with %d queries", len(baseline_results))

    # Run new eval
    logger.info("Running eval on new collection...")
    new_results, new_qdrant_count, new_pointsjsonl_count = run_eval()
    logger.info(
        "New eval complete: %d queries, %d qdrant points, %d jsonl points",
        len(new_results),
        new_qdrant_count,
        new_pointsjsonl_count,
    )

    # Build lookup
    new_by_qnum = {r["q_num"]: r for r in new_results}
    baseline_by_qnum = {r["q_num"]: r for r in baseline_results}

    # Classification
    classifications = classify_queries(new_results)

    # Split by type
    all_conceptual = [r for r in new_results if r["qtype"] == "conceptual"]
    low_leak = sorted(
        [r for r in all_conceptual if r["leakage"] <= LEAKAGE_THRESHOLD],
        key=lambda x: x["i"],
    )
    high_leak = [r for r in all_conceptual if r["leakage"] > LEAKAGE_THRESHOLD]
    exact_token = [r for r in new_results if r["qtype"] == "exact_token"]

    low_leak_qnums = {r["q_num"] for r in low_leak}

    # ──────────────────────────────────────────────
    # Section helpers
    # ──────────────────────────────────────────────

    def mrr_slice(results_slice, k=RRF_K):
        vals = [r["crr_by_k"][k] for r in results_slice]
        return statistics.mean(vals) if vals else 0.0

    def old_mrr_slice(qnums, k=RRF_K):
        vals = [
            baseline_by_qnum[q]["crr_by_k"][k] for q in qnums if q in baseline_by_qnum
        ]
        return statistics.mean(vals) if vals else 0.0

    # ──────────────────────────────────────────────
    # REPORT
    # ──────────────────────────────────────────────

    report = "# Anomaly Child Experiment Findings\n\n"
    report += f"Generated by `pipeline/compare_anomaly_child.py`\n\n"
    report += f"**Hypothesis:** Adding a separate child point containing raw (unmapped) ASRS anomaly text will improve retrieval MRR, especially on low-leakage queries.\n\n"
    report += "---\n\n"

    # ── Section 1: Overall ──
    report += "## Section 1: Overall Results\n\n"
    slices = [
        ("All queries", new_results),
        ("Conceptual (all)", all_conceptual),
        ("Exact_token", exact_token),
        ("Low-leakage conceptual", low_leak),
        ("High-leakage conceptual", high_leak),
    ]
    report += (
        "| Slice | n | Baseline MRR (k=60) | New MRR (k=60) | Δ MRR |\n"
        "|-------|---|---------------------|----------------|-------|\n"
    )
    for label, slice_data in slices:
        qnums = [r["q_num"] for r in slice_data]
        old_mrr = old_mrr_slice(qnums)
        new_mrr = mrr_slice(slice_data)
        delta = new_mrr - old_mrr
        report += f"| {label} | {len(slice_data)} | {old_mrr:.4f} | {new_mrr:.4f} | {delta:+.4f} |\n"
    report += "\n"

    # Per-query table (all 72)
    report += "### All 72 Queries — Per-Query Detail\n\n"
    report += (
        "| Q# | Type | Leak% | Old Comb Rank | New Comb Rank | Old MRR | New MRR | Δ MRR |\n"
        "|----|------|-------|--------------|--------------|---------|---------|-------|\n"
    )
    for r in new_results:
        b = baseline_by_qnum[r["q_num"]]
        old_rank = b["combined_gold_ranks"][RRF_K]
        new_rank = r["combined_gold_ranks"][RRF_K]
        old_mrr = b["crr_by_k"][RRF_K]
        new_mrr = r["crr_by_k"][RRF_K]
        delta = new_mrr - old_mrr
        report += f"| Q{r['q_num']} | {r['qtype'][:10]} | {100 * r['leakage']:.0f}% | {old_rank:>3} | {new_rank:>3} | {old_mrr:.4f} | {new_mrr:.4f} | {delta:+.4f} |\n"
    report += "\n---\n\n"

    # ── Section 2: Low-Leakage Deep Dive ──
    report += "## Section 2: Low-Leakage Deep Dive (15 queries)\n\n"
    report += "### Per-Query Results\n\n"
    report += (
        "| Q# | Query (truncated) | Class | Old Comb Rank | New Comb Rank | Old MRR | New MRR | Δ MRR | Helped? |\n"
        "|----|-------------------|-------|--------------|--------------|---------|---------|-------|--------|\n"
    )
    for r in low_leak:
        b = baseline_by_qnum[r["q_num"]]
        old_rank = b["combined_gold_ranks"][RRF_K]
        new_rank = r["combined_gold_ranks"][RRF_K]
        old_mrr = b["crr_by_k"][RRF_K]
        new_mrr = r["crr_by_k"][RRF_K]
        delta = new_mrr - old_mrr
        cls = classifications.get(r["q_num"], "?")
        helped = (
            "YES" if new_rank < old_rank else ("SAME" if new_rank == old_rank else "NO")
        )
        report += f"| Q{r['q_num']} | {r['query'][:55]} | {cls:>10} | {old_rank:>3} | {new_rank:>3} | {old_mrr:.4f} | {new_mrr:.4f} | {delta:+.4f} | {helped} |\n"
    report += "\n"

    # Aggregates
    report += "### Aggregate by Subgroup\n\n"
    report += "| Group | n | Baseline MRR (k=60) | New MRR (k=60) | Δ MRR |\n"
    report += "|-------|---|---------------------|----------------|-------|\n"

    answerable_qnums = [
        r["q_num"] for r in low_leak if classifications.get(r["q_num"]) == "ANSWERABLE"
    ]
    buried_qnums = [
        r["q_num"] for r in low_leak if classifications.get(r["q_num"]) == "BURIED"
    ]

    for label, qnums in [
        ("ANSWERABLE", answerable_qnums),
        ("BURIED", buried_qnums),
        ("Low-leakage total", [r["q_num"] for r in low_leak]),
    ]:
        old_mrr = old_mrr_slice(qnums)
        new_mrr = mrr_slice([r for r in low_leak if r["q_num"] in qnums])
        delta = new_mrr - old_mrr
        report += f"| {label} | {len(qnums)} | {old_mrr:.4f} | {new_mrr:.4f} | {delta:+.4f} |\n"
    report += "\n---\n\n"

    # ── Section 3: Edge Cases ──
    report += "## Section 3: Edge Cases (High-Leakage + Exact Token)\n\n"
    report += (
        "| Slice | n | Baseline MRR | New MRR | Δ MRR | Any regression? |\n"
        "|-------|---|-------------|---------|-------|----------------|\n"
    )
    edge_slices = [
        ("High-leakage conceptual", high_leak),
        ("Exact_token", exact_token),
    ]
    for label, slice_data in edge_slices:
        qnums = [r["q_num"] for r in slice_data]
        old_mrr = old_mrr_slice(qnums)
        new_mrr = mrr_slice(slice_data)
        delta = new_mrr - old_mrr
        any_regression = sum(
            1
            for r in slice_data
            if r["crr_by_k"][RRF_K] < baseline_by_qnum[r["q_num"]]["crr_by_k"][RRF_K]
        )
        report += f"| {label} | {len(slice_data)} | {old_mrr:.4f} | {new_mrr:.4f} | {delta:+.4f} | {any_regression}/{len(slice_data)} queries worsened |\n"
    report += "\n---\n\n"

    # ── Section 4: Q61 Spotlight ──
    q61_new = new_by_qnum.get(61)
    q61_old = baseline_by_qnum.get(61)
    if q61_new and q61_old:
        report += "## Section 4: Spotlight — Q61 (Landing Without Clearance)\n\n"
        report += f'Query: "{q61_new["query"]}"\n\n'
        report += "| Metric | Before | After | Δ |\n|--------|--------|-------|---|\n"
        report += f"| Dense rank | {q61_old['dense_gold_rank']} | {q61_new['dense_gold_rank']} | {q61_new['dense_gold_rank'] - q61_old['dense_gold_rank']:+,d} |\n"
        report += f"| Sparse rank | {q61_old['sparse_gold_rank']} | {q61_new['sparse_gold_rank']} | {q61_new['sparse_gold_rank'] - q61_old['sparse_gold_rank']:+,d} |\n"
        report += f"| Combined rank (k=60) | {q61_old['combined_gold_ranks'][RRF_K]} | {q61_new['combined_gold_ranks'][RRF_K]} | {q61_new['combined_gold_ranks'][RRF_K] - q61_old['combined_gold_ranks'][RRF_K]:+,d} |\n"
        report += f"| Combined MRR (k=60) | {q61_old['crr_by_k'][RRF_K]:.4f} | {q61_new['crr_by_k'][RRF_K]:.4f} | {q61_new['crr_by_k'][RRF_K] - q61_old['crr_by_k'][RRF_K]:+.4f} |\n"
        report += "\n"

        # Anomaly child similarity
        report += "**Anomaly child impact:**\n\n"
        # Compute cosine sim between query and anomaly child chunk
        q61_expected = q61_new["expected"]
        report += f"Gold ACN(s): {q61_expected}\n\n"
        report += f"**Verdict:** See rank/MRR delta above.\n\n"
    else:
        report += "## Section 4: Spotlight — Q61\n\nQ61 not found in results.\n\n"
    report += "---\n\n"

    # ── Section 5: Q70 Spotlight ──
    q70_new = new_by_qnum.get(70)
    q70_old = baseline_by_qnum.get(70)
    if q70_new and q70_old:
        report += "## Section 5: Spotlight — Q70 (Gear-Up Landing)\n\n"
        report += f'Query: "{q70_new["query"]}"\n\n'
        report += "| Metric | Before | After | Δ |\n|--------|--------|-------|---|\n"
        report += f"| Dense rank | {q70_old['dense_gold_rank']} | {q70_new['dense_gold_rank']} | {q70_new['dense_gold_rank'] - q70_old['dense_gold_rank']:+,d} |\n"
        report += f"| Sparse rank | {q70_old['sparse_gold_rank']} | {q70_new['sparse_gold_rank']} | {q70_new['sparse_gold_rank'] - q70_old['sparse_gold_rank']:+,d} |\n"
        report += f"| Combined rank (k=60) | {q70_old['combined_gold_ranks'][RRF_K]} | {q70_new['combined_gold_ranks'][RRF_K]} | {q70_new['combined_gold_ranks'][RRF_K] - q70_old['combined_gold_ranks'][RRF_K]:+,d} |\n"
        report += f"| Combined MRR (k=60) | {q70_old['crr_by_k'][RRF_K]:.4f} | {q70_new['crr_by_k'][RRF_K]:.4f} | {q70_new['crr_by_k'][RRF_K] - q70_old['crr_by_k'][RRF_K]:+.4f} |\n"
    else:
        report += "## Section 5: Spotlight — Q70\n\nQ70 not found in results.\n\n"
    report += "\n---\n\n"

    # ── Section 6: False-Positive Audit ──
    report += "## Section 6: False-Positive Audit\n\n"
    report += "For each low-leakage query, comparing top-5 combined results before vs after.\n"
    report += "Flagging any non-gold ACN that entered top-5 in the new system.\n\n"

    for r in low_leak:
        qnum = r["q_num"]
        b = baseline_by_qnum.get(qnum)
        if not b:
            continue
        expected = set(r["expected"])
        old_top5 = {item["acn"] for item in b["top10_combined"][:5]}
        new_top5 = {item["acn"] for item in r["top10_combined"][:5]}
        new_entries = new_top5 - old_top5
        non_gold_new = [acn for acn in new_entries if acn not in expected]
        gold_moved = bool(new_top5 & expected and not (old_top5 & expected))

        report += f'**Q{qnum}: "{r["query"][:60]}"**\n'
        report += f"- Expected ACN(s): {sorted(expected)}\n"
        report += f"- Old top-5: {[item['acn'] for item in b['top10_combined'][:5]]}\n"
        report += f"- New top-5: {[item['acn'] for item in r['top10_combined'][:5]]}\n"
        if non_gold_new:
            report += f"- ⚠️  Non-gold ACNs that entered top-5: {non_gold_new}\n"
            report += f"  - Classification: Needs manual inspection to determine if legitimate or spurious.\n"
        else:
            report += f"- ✅ No new non-gold entries in top-5.\n"
        if gold_moved:
            report += f"- ✅ Gold ACN entered top-5 (was absent before).\n"
        report += "\n"

    report += "---\n\n"

    # ── Section 7: Verdict ──
    report += "## Section 7: Verdict\n\n"

    # Compute stats
    ll_qnums = [r["q_num"] for r in low_leak]
    old_ll_mrr = old_mrr_slice(ll_qnums)
    new_ll_mrr = mrr_slice(low_leak)
    ll_delta = new_ll_mrr - old_ll_mrr

    n_improved = sum(
        1
        for r in low_leak
        if r["combined_gold_ranks"][RRF_K]
        < baseline_by_qnum[r["q_num"]]["combined_gold_ranks"][RRF_K]
    )
    n_worsened = sum(
        1
        for r in low_leak
        if r["combined_gold_ranks"][RRF_K]
        > baseline_by_qnum[r["q_num"]]["combined_gold_ranks"][RRF_K]
    )
    n_same = len(low_leak) - n_improved - n_worsened

    hl_qnums = [r["q_num"] for r in high_leak]
    hl_regressed = sum(
        1
        for r in high_leak
        if r["crr_by_k"][RRF_K] < baseline_by_qnum[r["q_num"]]["crr_by_k"][RRF_K]
    )
    et_regressed = sum(
        1
        for r in exact_token
        if r["crr_by_k"][RRF_K] < baseline_by_qnum[r["q_num"]]["crr_by_k"][RRF_K]
    )

    points_delta = new_qdrant_count - baseline_point_count

    report += f"### 1. Did the anomaly child improve low-leakage MRR?\n\n"
    report += f"**Low-leakage MRR (k=60): {old_ll_mrr:.4f} → {new_ll_mrr:.4f} (Δ={ll_delta:+.4f})**\n\n"
    if ll_delta > 0.01:
        report += f"Yes, meaningful improvement of {ll_delta:+.4f} MRR points.\n\n"
    elif ll_delta > 0:
        report += f"Marginal improvement of {ll_delta:+.4f} MRR points.\n\n"
    else:
        report += f"No improvement (Δ={ll_delta:+.4f}).\n\n"

    report += f"### 2. Which queries drove the gain vs which were neutral/harmed?\n\n"
    report += f"- Improved combined rank: {n_improved}/{len(low_leak)}\n"
    report += f"- Worsened combined rank: {n_worsened}/{len(low_leak)}\n"
    report += f"- Unchanged: {n_same}/{len(low_leak)}\n\n"

    # Detail which improved
    improved_list = [
        r["q_num"]
        for r in low_leak
        if r["combined_gold_ranks"][RRF_K]
        < baseline_by_qnum[r["q_num"]]["combined_gold_ranks"][RRF_K]
    ]
    worsened_list = [
        r["q_num"]
        for r in low_leak
        if r["combined_gold_ranks"][RRF_K]
        > baseline_by_qnum[r["q_num"]]["combined_gold_ranks"][RRF_K]
    ]
    if improved_list:
        report += (
            f"**Improved:** Q{', Q'.join(str(q) for q in sorted(improved_list))}\n\n"
        )
    if worsened_list:
        report += (
            f"**Worsened:** Q{', Q'.join(str(q) for q in sorted(worsened_list))}\n\n"
        )

    report += f"### 3. Did it hurt any non-low-leakage queries?\n\n"
    total_edge = len(high_leak) + len(exact_token)
    total_regressed = hl_regressed + et_regressed
    report += f"- High-leakage conceptual: {hl_regressed}/{len(high_leak)} queries regressed\n"
    report += f"- Exact_token: {et_regressed}/{len(exact_token)} queries regressed\n"
    if total_regressed == 0:
        report += "**No regression detected in non-low-leakage slices.**\n\n"
    else:
        report += (
            f"**{total_regressed}/{total_edge} non-low-leakage queries regressed.**\n\n"
        )

    report += f"### 4. Any false positives?\n\n"
    total_new_entries = 0
    for r in low_leak:
        b = baseline_by_qnum.get(r["q_num"])
        if not b:
            continue
        expected = set(r["expected"])
        old_top5 = {item["acn"] for item in b["top10_combined"][:5]}
        new_top5 = {item["acn"] for item in r["top10_combined"][:5]}
        new_entries = (new_top5 - old_top5) - expected
        total_new_entries += len(new_entries)
    report += f"Non-gold ACNs entering top-5 across low-leakage queries: {total_new_entries}\n"
    report += "See Section 6 for per-query details. Manual inspection recommended for classification.\n\n"

    report += f"### 5. Storage cost\n\n"
    report += f"- Baseline Qdrant pool points: {baseline_point_count}\n"
    report += f"- New Qdrant pool points: {new_qdrant_count}\n"
    report += f"- Points added to pool: +{new_qdrant_count - baseline_point_count}\n"
    report += f"- Full JSONL: ~43,847 → {new_pointsjsonl_count} (+{new_pointsjsonl_count - 43847:,}, of which +{new_qdrant_count - baseline_point_count} are in the 250-ACN pool)\n\n"

    report += f"### 6. Was it a good choice?\n\n"
    if ll_delta > 0.01 and total_regressed <= 2:
        report += f"**YES.** "
        report += f"The anomaly child improved low-leakage MRR by {ll_delta:+.4f} "
        report += (
            f"({n_improved}/{len(low_leak)} queries improved, {n_worsened} worsened) "
        )
        if total_regressed == 0:
            report += "with zero regression in non-low-leakage slices. "
        else:
            report += f"with minimal regression ({total_regressed}/{total_edge} edge queries). "
        report += f"The storage cost was {new_pointsjsonl_count - baseline_point_count:,} additional points "
        report += (
            f"({new_pointsjsonl_count - baseline_point_count:,} in the full dataset, "
        )
        report += f"{new_qdrant_count - baseline_point_count} in the pool). "
        report += "The anomaly child approach works because it bypasses the ADREP mapping dilution entirely, "
        report += (
            "delivering 100% unmapped anomaly signal directly to the dense retriever."
        )
    elif ll_delta > 0:
        report += f"**BORDERLINE.** "
        report += f"The improvement is marginal ({ll_delta:+.4f}) and may not justify the added complexity. "
        report += f"Consider alternative approaches like query expansion or structured pre-filtering."
    else:
        report += f"**NO.** "
        report += (
            f"The anomaly child did not improve low-leakage MRR (Δ={ll_delta:+.4f}). "
        )
        report += "The hypothesis was not confirmed by the data."

    with open("anomaly_child_findings.md", "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"REPORT WRITTEN TO anomaly_child_findings.md")
    print(f"{'=' * 60}")
    print(f"  Low-leak MRR: {old_ll_mrr:.4f} → {new_ll_mrr:.4f} (Δ={ll_delta:+.4f})")
    print(f"  Improved: {n_improved}, Worsened: {n_worsened}, Same: {n_same}")
    print(
        f"  Qdrant points: {baseline_point_count} → {new_qdrant_count} (+{new_qdrant_count - baseline_point_count})"
    )


if __name__ == "__main__":
    main()
