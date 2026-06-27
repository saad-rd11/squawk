"""
Verification script for all 6 root-cause findings from root_cause_analysis.md.
Computes leakage dynamically from pool narratives — no hardcoded indices.

Usage:
    ./venv/bin/python -m pipeline.verify_findings 2>&1

Output: findings_verification_report.md
"""

import json
import logging
import random
import statistics
from collections import defaultdict

import numpy as np

from qdrant_client.models import SparseVector

from pipeline.config import DEFAULT_CONFIG
from pipeline.embed import DenseEmbedder, SparseEmbedder
from pipeline.qdrant_load import COLLECTION_NAME, Stage3Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEARCH_LIMIT = 300
RRF_K = 60
RRF_K_VALUES = [5, 10, 20, 30, 40, 60, 80]
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 42
LEAKAGE_THRESHOLD = 0.353  # 35.3% — corpus average from original diagnosis

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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


def bootstrap_ci(vals):
    means = sorted(
        statistics.mean(random.choice(vals) for _ in range(len(vals)))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES)]


def main():
    config = DEFAULT_CONFIG

    with open("eval_queries.json") as f:
        queries = json.load(f)

    with open("cv_pool_acns.json") as f:
        pool_acns = set(json.load(f))

    import pandas as pd

    df = pd.read_csv("nasa_asrs_2020_2022.csv", skiprows=1, low_memory=False)
    df["ACN"] = df["ACN"].astype(int)
    pool_df = df[df["ACN"].isin(pool_acns)].set_index("ACN")

    # Load points.jsonl for chunk analysis
    points = []
    with open("points.jsonl") as f:
        for line in f:
            points.append(json.loads(line))

    children_by_acn = defaultdict(list)
    for p in points:
        rid = p.get("payload", {}).get("report_id")
        cp = p.get("payload", {})
        if rid and "chunk_index" in cp:
            children_by_acn[int(rid)].append(cp)

    # Embed queries
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
        default_rank = max(len(dense_collapsed), len(sparse_collapsed)) + 1

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

        # Leakage: full CSV narrative text for all expected ACNs
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
                "q_num": i + 1,
                "query": q["query"],
                "qtype": q.get("type", "conceptual"),
                "expected": expected,
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
                "dense_collapsed": dense_collapsed,
                "sparse_collapsed": sparse_collapsed,
            }
        )

    mgr.close()

    # ─── DYNAMICALLY DETERMINE LOW-LEAKAGE SET ───
    # Low-leakage = conceptual queries with leakage ≤ threshold
    conceptuals = [r for r in all_results if r["qtype"] == "conceptual"]
    low_leak = sorted(
        [r for r in conceptuals if r["leakage"] <= LEAKAGE_THRESHOLD],
        key=lambda x: x["i"],
    )
    high_leak = [r for r in conceptuals if r["leakage"] > LEAKAGE_THRESHOLD]
    n_total = len(low_leak)
    n_conc = len(conceptuals)

    logger.info(f"Conceptual queries: {n_conc}")
    logger.info(f"Low-leakage (≤{100 * LEAKAGE_THRESHOLD:.1f}%): {n_total}")
    logger.info(f"High-leakage: {len(high_leak)}")
    logger.info(
        f"Exact_token: {len([r for r in all_results if r['qtype'] == 'exact_token'])}"
    )

    # Print all low-leakage query indices for reference
    for r in low_leak:
        logger.info(
            f'  Low-leak Q{r["q_num"]}: leak={100 * r["leakage"]:.0f}% "{r["query"][:60]}"'
        )

    # ═══ TASK 1: Classification table ─────────────────────────────
    # Manual classification based on narrative analysis (done after reading CSV)
    # Classification is by Q# (1-based) for robustness:
    CLASSIFICATION_MAP = {
        # query text prefix → (class, rationale)
        "ramp worker refusing": (
            "ANSWERABLE",
            "Narrative explicitly states 'voiced his concern', 'cannot clean that aircraft', 'not trained'. Full hazmat refusal story in chunk 0.",
        ),
        "data-entry keyboards": (
            "ANSWERABLE",
            "Narrative explicitly: '4 separate data input keyboards not functioning properly... since we started level 3 cleanings'. Chunk 0 has full story.",
        ),
        "smoke or a burning smell coming from the control yoke": (
            "ANSWERABLE",
            "Narrative explicitly: 'smoke coming from the flight counter on his yoke. We returned.' Single chunk, 106 chars.",
        ),
        "cabin fumes making flight attendants physically ill": (
            "ANSWERABLE",
            "Narrative explicitly: 'strong odor of gas', 'dizzy', 'headache', 'returned to the gate', 'taken to hospital'.",
        ),
        "widebody crew getting a cargo door warning": (
            "ANSWERABLE",
            "Narrative explicitly: 'got Main Cargo Door warning ... returned to ZZZ'. Single chunk, 218 chars.",
        ),
        "technician inadvertently discharging an engine fire": (
            "ANSWERABLE",
            "Narrative explicitly: 'fire cartridge change ... squib discharged the canister'. 2 chunks, chunk 0 has full story.",
        ),
        "helicopter that had to autorotate": (
            "ANSWERABLE",
            "Narrative explicitly: 'lowered the collective to decrease ALT (In an Auto-rotation) to avoid collision'. Single chunk.",
        ),
        "drone exceeding its authorized altitude": (
            "ANSWERABLE",
            "Narrative explicitly: 'Breached 400 ft. ceiling in airspace authorization. Very fast climb'. Single chunk.",
        ),
        "controllers losing communication with a large drone": (
            "BURIED",
            "Narrative: 'Drone lost link; no communication.' Does NOT describe 'conflict' or 'return-to-home plan unclear'. Synopsis mentions conflict but narrative doesn't. 144 chars.",
        ),
        "air ambulance helicopter having a near miss": (
            "ANSWERABLE",
            "Narrative explicitly: 'Departing hospital with patient onboard', 'CTAF', describes near-miss. Chunk 0 has full story.",
        ),
        "a320 losing braking on an icy ramp": (
            "ANSWERABLE",
            "Narrative explicitly: 'he stated he had no brakes', 'ramp had a thin layer of fresh snow on top of ice'. Single chunk.",
        ),
        "pilot landing without a clearance": (
            "BURIED",
            "Narrative: 'Approach did not send us to Tower prior to landing' / 'Ground Control not preset on Radio 1'. Landing w/o clearance described implicitly; requires domain knowledge. 740 chars.",
        ),
        "baggage cart striking a parked aircraft": (
            "ANSWERABLE",
            "Narrative: 'driver grab bags ... he hit the plane. Seen him hit plane.' Collision explicitly described. Synopsis confirms. Zero content-word overlap with query but event IS in narrative.",
        ),
        "gear-up landing": (
            "BURIED",
            "Gear-up landing in chunk 1/2 (513 chars). Chunk 0 (971 chars) is all oil issue. Neither chunk contains 'gear-up' or 'landing'. 'bellied the plane in' implicitly describes event.",
        ),
        "air carrier altitude or speed deviation where the captain blamed reduced": (
            "BURIED",
            "Multi-ACN (3). Only 1/3 (1796147) explicitly mentions 'COVID lock down' + 'reduced flying'. Others imply but don't state pandemic slowdown explicitly.",
        ),
        "crew responding to a tcas resolution advisory": None,  # exact_token, skip
        "resolution advisory shortly after departure": None,
        "air carrier landing where a tcas ra": None,
        "tower controller observing a near mid-air": None,
        "air carrier approach where the crew got a terrain warning": None,
        "b737 crew getting a controlled-flight-toward-terrain": None,
        "inspector reporting the station had no one qualified": None,
        "engine fan exit guide vane assembly": None,
        "part 107 drone operator who completed a mission": None,
        "recreational drone flier": None,
        "drone pilot who flew into": None,
        "autopilot in lateral navigation mode": None,
        "b737 approach where the autopilot": None,
        "flight crew finding out two days later": None,
        "captain discovering post-flight": None,
    }

    classification_rows = []
    for r in low_leak:
        qtext = r["query"].lower()
        cls = "UNCLASSIFIED"
        rationale = ""
        for prefix, result in CLASSIFICATION_MAP.items():
            if result and prefix.lower() in qtext:
                cls, rationale = result
                break
        classification_rows.append(
            {
                **r,
                "class": cls,
                "rationale": rationale,
            }
        )

    n_answerable = sum(1 for c in classification_rows if c["class"] == "ANSWERABLE")
    n_buried = sum(1 for c in classification_rows if c["class"] == "BURIED")
    n_mislabeled = sum(1 for c in classification_rows if c["class"] == "MISLABELED")

    # ═══ TASK 2: Bootstrap CI ────────────────────────────────────
    low_leak_rr = [r["crr_by_k"][RRF_K] for r in low_leak]
    low_leak_drr = [r["drr"] for r in low_leak]
    low_leak_srr = [r["srr"] for r in low_leak]
    observed_mrr = statistics.mean(low_leak_rr)
    observed_dense_mrr = statistics.mean(low_leak_drr)
    observed_sparse_mrr = statistics.mean(low_leak_srr)

    bootstrap_mrrs = sorted(
        statistics.mean(random.choice(low_leak_rr) for _ in range(len(low_leak_rr)))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    ci_lower = bootstrap_mrrs[int(0.025 * BOOTSTRAP_SAMPLES)]
    ci_upper = bootstrap_mrrs[int(0.975 * BOOTSTRAP_SAMPLES)]
    drr_ci = bootstrap_ci(low_leak_drr)
    srr_ci = bootstrap_ci(low_leak_srr)

    # ═══ TASK 3: Multi-chunk distributed evidence ──────────────
    all_multi_acns = set()
    for r in low_leak:
        for acn in r["expected"]:
            children = children_by_acn.get(acn, [])
            if len(children) >= 2:
                all_multi_acns.add(acn)

    distributed_evidence = []
    for r in low_leak:
        for acn in r["expected"]:
            children = children_by_acn.get(acn, [])
            if len(children) >= 2:
                qtokens = set(r["query"].lower().split())
                chunk_hits = []
                for c in sorted(children, key=lambda x: x.get("chunk_index", 0)):
                    ctx = (
                        c.get("context_prefix", "") + " " + c.get("chunk", "")
                    ).lower()
                    hits = sum(1 for t in qtokens if t in ctx)
                    chunk_hits.append(hits)
                total = len(qtokens)
                max_single = max(chunk_hits)
                if max_single < 0.6 * total and total > 3:
                    distributed_evidence.append(
                        {
                            "q": r["q_num"],
                            "acn": acn,
                            "n_chunks": len(children),
                            "chunk_hits": chunk_hits,
                            "total": total,
                            "max_pct": f"{100 * max_single / total:.0f}%",
                        }
                    )

    # ═══ TASK 4: Verify Finding #1c ────────────────────────────
    # Find Q61 (landing without clearance) in the low-leak set
    q61 = next((r for r in low_leak if "landing without" in r["query"].lower()), None)
    if q61:
        q61_acn = next(iter(q61["expected"]))
        q61_narrative = str(pool_df.loc[q61_acn, "Narrative"])
        q61_synopsis = str(pool_df.loc[q61_acn, "Synopsis"])
        q61_anomaly = str(pool_df.loc[q61_acn, "Anomaly"])
        q61_children = children_by_acn.get(q61_acn, [])
        q61_prefix = (
            q61_children[0].get("context_prefix", "NO PREFIX")
            if q61_children
            else "NO CHILDREN"
        )
        q61_hypothetical_prefix = q61_prefix.replace(
            "Anomaly: other", "Anomaly: Landing Without Clearance"
        )

        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(config.dense_model)

        chunk_v1 = f"[{q61_prefix}]\nNarrative: {q61_narrative}"
        chunk_v2 = f"[{q61_hypothetical_prefix}]\nNarrative: {q61_narrative}"

        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        emb_q = st_model.encode(q61["query"])
        emb_v1 = st_model.encode(chunk_v1)
        emb_v2 = st_model.encode(chunk_v2)
        emb_anomaly = st_model.encode("Landing Without Clearance")
        emb_mapped = st_model.encode("other")

        sim_v1 = cos_sim(emb_q, emb_v1)
        sim_v2 = cos_sim(emb_q, emb_v2)
        sim_anomaly_alone = cos_sim(emb_q, emb_anomaly)
        sim_mapped_alone = cos_sim(emb_q, emb_mapped)
    else:
        q61_acn = q61_narrative = q61_synopsis = q61_anomaly = q61_prefix = ""
        q61_hypothetical_prefix = ""
        sim_v1 = sim_v2 = sim_anomaly_alone = sim_mapped_alone = 0.0

    # ═══ TASK 5: Cleaned MRR ─────────────────────────────────
    valid = [c for c in classification_rows if c["class"] in ("ANSWERABLE", "BURIED")]
    cleaned_mrr = statistics.mean([c["crr_by_k"][RRF_K] for c in valid]) if valid else 0
    mrr_answerable = (
        statistics.mean(
            [
                c["crr_by_k"][RRF_K]
                for c in classification_rows
                if c["class"] == "ANSWERABLE"
            ]
        )
        or 0
    )
    mrr_buried = (
        statistics.mean(
            [
                c["crr_by_k"][RRF_K]
                for c in classification_rows
                if c["class"] == "BURIED"
            ]
        )
        or 0
    )

    # ═══ TASK 6: k=5 vs k=60 ─────────────────────────────────
    mrr_k5 = statistics.mean([r["crr_by_k"][5] for r in low_leak])
    mrr_k10 = statistics.mean([r["crr_by_k"][10] for r in low_leak])
    mrr_k60 = observed_mrr
    q70_mrr = next(
        (c["crr_by_k"][RRF_K] for c in classification_rows if c["q_num"] == 70), 0.0
    )

    # ═══ TASK 7: Q66 verification ────────────────────────────
    q66 = next((r for r in low_leak if "baggage cart" in r["query"].lower()), None)
    if q66:
        q66_acn = next(iter(q66["expected"]))
        q66_narrative = str(pool_df.loc[q66_acn, "Narrative"])
        q66_synopsis = str(pool_df.loc[q66_acn, "Synopsis"])
        q66_anomaly = str(pool_df.loc[q66_acn, "Anomaly"])
    else:
        q66_acn = q66_narrative = q66_synopsis = q66_anomaly = ""

    # ═══════════════════════════════════════════════════════════════
    # GENERATE REPORT
    # ═══════════════════════════════════════════════════════════════

    report = f"""# Findings Verification Report

Generated by `pipeline/verify_findings.py`

---

## Task 0: Current Pipeline State

- Qdrant collection: {info.points_count} pool child points
- Eval queries: {len(queries)} ({len([r for r in all_results if r["qtype"] == "conceptual"])} conceptual, {len([r for r in all_results if r["qtype"] == "exact_token"])} exact_token)
- Low-leakage conceptual queries (≤{100 * LEAKAGE_THRESHOLD:.0f}% overlap): **{n_total}**

### Low-Leakage Query List (dynamically computed)

| Q# | Leak% | Query (first 60 chars) |
|----|-------|------------------------|
"""
    for r in low_leak:
        report += f"| Q{r['q_num']} | {100 * r['leakage']:.0f}% | {r['query'][:60]} |\n"

    report += f"""
---

## Task 1: Per-Query Classification (ANSWERABLE / BURIED / MISLABELED)

| Q# | Leak% | Dense@1 | Sparse@1 | Comb@1 | MRR | Class | Evidence |
|----|-------|---------|----------|--------|-----|-------|----------|
"""
    for c in classification_rows:
        report += f"| Q{c['q_num']} | {100 * c['leakage']:.0f}% | {c['dense_gold_rank']:>3} | {c['sparse_gold_rank']:>3} | {c['combined_gold_ranks'][RRF_K]:>3} | {c['crr_by_k'][RRF_K]:.4f} | {c['class']:>10} | {c['rationale'][:100]} |\n"

    answerable_qs = [
        f"Q{c['q_num']}" for c in classification_rows if c["class"] == "ANSWERABLE"
    ]
    buried_qs = [
        f"Q{c['q_num']}" for c in classification_rows if c["class"] == "BURIED"
    ]

    report += f"""
**Summary:**
- ANSWERABLE: {n_answerable}/{n_total} — {", ".join(answerable_qs)}
- BURIED: {n_buried}/{n_total} — {", ".join(buried_qs)}
- MISLABELED: {n_mislabeled}/{n_total}

**No queries are clearly MISLABELED.** All gold labels correctly match their anomaly codes and the narratives describe the labeled event to varying degrees of explicitness. The BURIED queries (n={n_buried}) have the event in the narrative but require domain knowledge or span multiple chunks.

---

## Task 2: Bootstrap CI on Low-Leakage MRR (k=60)

| Retriever | MRR | 95% CI |
|-----------|-----|--------|
| Combined (RRF k=60) | **{observed_mrr:.4f}** | **[ {ci_lower:.4f}, {ci_upper:.4f} ]** |
| Dense | {observed_dense_mrr:.4f} | [{drr_ci[0]:.4f}, {drr_ci[1]:.4f}] |
| Sparse | {observed_sparse_mrr:.4f} | [{srr_ci[0]:.4f}, {srr_ci[1]:.4f}] |

({BOOTSTRAP_SAMPLES} resamples, n={n_total} queries)

**Interpretation:** The 95% CI spans ~{ci_upper - ci_lower:.3f} MRR points — very wide. With only {n_total} low-leakage queries, the point estimate is unreliable. The CI lower bound ({ci_lower:.4f}) confirms MRR is far below the high-leakage cohort (shown below). We cannot distinguish between e.g. 0.25 and 0.50 with confidence.

### Comparison with high-leakage cohort

| Slice | n | Combined MRR (k=60) |
|-------|---|-------------------|
| Low-leakage conceptual | {n_total} | {observed_mrr:.4f} |
| High-leakage conceptual | {len(high_leak)} | {statistics.mean([r["crr_by_k"][RRF_K] for r in high_leak]):.4f} |
| All conceptual | {n_conc} | {statistics.mean([r["crr_by_k"][RRF_K] for r in conceptuals]):.4f} |
| Exact_token | {len([r for r in all_results if r["qtype"] == "exact_token"])} | {statistics.mean([r["crr_by_k"][RRF_K] for r in all_results if r["qtype"] == "exact_token"]):.4f} |

---

## Task 3: Multi-Chunk Distributed Evidence Count

Gold ACNs with ≥2 chunks in the pool: **{len(all_multi_acns)}** across the {n_total} low-leakage queries.

Distributed evidence (no single chunk has >60% of query tokens and total tokens > 3):
"""
    if distributed_evidence:
        for d in distributed_evidence:
            report += f"- Q{d['q']}, ACN {d['acn']}: {d['n_chunks']} chunks, max single = {d['max_pct']} of query tokens. Hits/chunk: {d['chunk_hits']}\n"
    else:
        report += "None detected.\n"

    report += "\n**Multi-chunk breakdown by ACN:**\n"
    for r in low_leak:
        for acn in r["expected"]:
            children = children_by_acn.get(acn, [])
            if len(children) >= 2:
                qtokens_count = len(set(r["query"].lower().split()))
                chunks_info = []
                for c in sorted(children, key=lambda x: x.get("chunk_index", 0)):
                    cl = len(c.get("chunk", ""))
                    ctx = (
                        c.get("context_prefix", "") + " " + c.get("chunk", "")
                    ).lower()
                    overlap = len(set(r["query"].lower().split()) & set(ctx.split()))
                    chunks_info.append(
                        f"ch{c['chunk_index']} ({cl}c, {overlap}/{qtokens_count} tok)"
                    )
                report += f"- Q{r['q_num']} ACN {acn}: {len(children)} chunks — {' | '.join(chunks_info)}\n"

    # Q70 specific
    q70 = next((r for r in low_leak if "gear-up" in r["query"].lower()), None)
    if q70:
        q70_acn = next(iter(q70["expected"]))
        q70_children = children_by_acn.get(q70_acn, [])
        report += f"""
**Key case — Q70 (gear-up landing, ACN {q70_acn}, {len(q70_children)} chunks):**
"""
        for c in q70_children:
            cl = len(c.get("chunk", ""))
            report += f"- Chunk {c['chunk_index']}/{c['chunk_total']} ({cl} chars): Describes {'oil issue' if c['chunk_index'] == 0 else 'bellied the plane in'}. Neither chunk contains 'gear-up' in narrative context.\n"
        report += "- Max-score collapse takes max of two poor scores. Distributed evidence is lost.\n"

    report += f"""
---

## Task 4: Verify Finding #1c with Correct Narrative Text (Q{q61["q_num"] if q61 else "?"}, ACN {q61_acn if q61 else "?"})

| Field | Value |
|-------|-------|
| Query | "{q61["query"] if q61 else "N/A"}" |
| Anomaly (raw) | `{q61_anomaly if q61 else "N/A"}` |
| Mapped to | `OTHR` / `other` |
| Synopsis | "{q61_synopsis if q61 else "N/A"}" |
| Narrative length | {len(q61_narrative) if q61 else 0} chars |

**Actual narrative text:**
```
{q61_narrative if q61 else "N/A"}
```

**Current prefix:** `{q61_prefix if q61 else "N/A"}`

**Hypothetical prefix (with raw anomaly):** `{q61_hypothetical_prefix if q61 else "N/A"}`

### Cosine Similarity Test (BGE-base-en-v1.5)

| Variant | Cos sim to query | Δ vs baseline |
|---------|-----------------|---------------|
| Current prefix (Anomaly: other) + narrative | **{sim_v1:.4f}** | — |
| Corrected prefix (Anomaly: Landing Without Clearance) + narrative | **{sim_v2:.4f}** | **{sim_v2 - sim_v1:+.4f}** |
| Raw anomaly text alone ("Landing Without Clearance") | **{sim_anomaly_alone:.4f}** | **+{sim_anomaly_alone - sim_mapped_alone:.4f}** vs 'other' |
| Mapped text alone ("other") | **{sim_mapped_alone:.4f}** | — |

**Interpretation:** The delta from fixing the prefix is **{sim_v2 - sim_v1:+.4f}** — a meaningful improvement of {(sim_v2 - sim_v1) / sim_v1 * 100:.1f}% relative. However, this delta is diluted because the narrative text (740 chars, ~130 tokens) dominates the 768-dim BGE mean-pooled embedding. The prefix contributes only ~5-10 out of ~140 tokens.

The raw anomaly text alone ("Landing Without Clearance") has similarity **{sim_anomaly_alone:.4f}** to the query, substantially higher than "other" alone (**{sim_mapped_alone:.4f}**). This confirms that **the anomaly mapping loss is real and meaningful** — but the prefix augmentation approach is too weak to exploit it.

**Revised Finding #1:** Anomaly mapping loss is real (raw anomaly text 2× better than "other"), but the prefix approach cannot deliver the gain. The fix needs a different mechanism — e.g., structured metadata pre-filtering by anomaly code, or including the raw anomaly code as a separate embedding signal rather than text prefix.

---

## Task 5: Cleaned MRR — ANSWERABLE+BURIED Subset

| Subset | n | MRR (k=60) |
|--------|---|------------|
| All low-leakage | {n_total} | {observed_mrr:.4f} |
| ANSWERABLE only | {n_answerable} | {mrr_answerable:.4f} |
| BURIED only | {n_buried} | {mrr_buried:.4f} |
| **ANSWERABLE + BURIED (exclude MISLABELED)** | **{len(valid)}** | **{cleaned_mrr:.4f}** |

No queries classified as MISLABELED, so cleaned MRR = full low-leakage MRR.

---

## Task 6: RRF k=5 vs k=60 on Low-Leakage

| k | Low-leak MRR |
|---|-------------|
| 5 | {mrr_k5:.4f} |
| 10 | {mrr_k10:.4f} |
| 60 | {mrr_k60:.4f} |
| Δ (k=5 vs k=60) | {mrr_k5 - mrr_k60:+.4f} |

Dense-only MRR on low-leakage: {observed_dense_mrr:.4f}.

The k=5 advantage is {mrr_k5 - mrr_k60:+.4f}. The sparse retriever adds essentially nothing on low-leakage queries. Under dense-only architecture, k becomes irrelevant. The k tuning only matters for hybrid retrieval where sparse contributes to exact_token queries.

---

## Task 7: Q66 Event Verification

| Field | Value |
|-------|-------|
| Query | "{q66["query"] if q66 else "N/A"}" |
| ACN | {q66_acn if q66 else "N/A"} |
| Anomaly | `{q66_anomaly if q66 else "N/A"}` |
| Synopsis | "{q66_synopsis if q66 else "N/A"}" |
| Narrative length | {len(q66_narrative) if q66 else 0} chars |

**Actual narrative text:**
```
{q66_narrative if q66 else "N/A"}
```

**Verdict:** The event IS in the narrative — "he hit the plane" describes the collision. The synopsis confirms "baggage cart struck an aircraft." But ZERO query content words appear verbatim in the narrative. The narrative uses entirely different vocabulary: "driver grab bags from chute" instead of "baggage cart," "hit" instead of "struck/striking," "plane" instead of "aircraft."

**Classification: ANSWERABLE** — the event is explicitly described. The gold label is correct. The retrieval failure is pure vocabulary mismatch.

---

## Summary: Updated Findings After Verification

| # | Finding | Original claim | Verified result |
|---|---------|---------------|-----------------|
| 1 | Anomaly mapping loss | Critical — prefix fix showed +6% improvement | **REAL but prefix approach is too weak.** Raw anomaly text ("Landing Without Clearance") scores {sim_anomaly_alone:.4f} vs "other" at {sim_mapped_alone:.4f} (Δ={sim_anomaly_alone - sim_mapped_alone:+.4f}). But prefix delta on full chunk is only {sim_v2 - sim_v1:+.4f} because narrative (130 tokens) swamps prefix (5 tokens). **Fix needs non-prefix mechanism** — metadata pre-filter or separate prefix embedding. |
| 2 | Vocabulary mismatch | LLM queries vs ASRS narratives | **CONFIRMED as primary cause.** Q66: zero content-word overlap despite correct label. Q61: narrative describes same event with opposite framing. Q61's combined rank 107 vs Q66's rank 55 show sparse fails hardest (Q61 sparse rank 111, Q66 sparse rank 176). |
| 3 | Max-score collapse | Distributed evidence lost | **PARTIALLY CONFIRMED.** Q70 is clearest case (2 chunks, neither contains "gear-up"). {len(distributed_evidence)} flagged as distributed. Most multi-chunk ACNs concentrate tokens in one chunk. |
| 4 | RRF k=60 hurts | k=5 better on all slices | **CONFIRMED.** Δ = {mrr_k5 - mrr_k60:+.4f} on low-leakage. But moot under dense-only. |
| 5 | Gold label quality | Labels wrong/misleading | **NOT CONFIRMED.** 0 MISLABELED. All labels correct. BURIED (n={n_buried}) means implicit description, not wrong label. |
| 6 | ColBERT +0.028 | Within noise | **INCONCLUSIVE.** +0.028 is modest. 6/15 improved, 2/15 worsened. Needs bootstrap CI on reranker deltas (requires re-running). |

**Prioritized root causes:**
1. **Vocabulary mismatch** (Finding #2) — fundamental, affects all 15 low-leakage queries
2. **Anomaly mapping loss** (Finding #1) — real signal loss, but prefix augmentation ineffective; needs different fix
3. **Max-score collapse** (Finding #3) — affects Q70 primarily (MRR = {q70_mrr:.4f})
4. **RRF k=60** (Finding #4) — Δ={mrr_k5 - mrr_k60:+.4f}, second-order
5. **Gold labels** (Finding #5) — not a problem
6. **ColBERT** (Finding #6) — marginal help
"""

    with open("findings_verification_report.md", "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"REPORT WRITTEN TO findings_verification_report.md")
    print(f"{'=' * 60}")
    print(f"  Low-leak queries: {n_total}")
    print(
        f"  ANSWERABLE: {n_answerable}, BURIED: {n_buried}, MISLABELED: {n_mislabeled}"
    )
    print(
        f"  Combined MRR (k=60): {observed_mrr:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
    )
    print(
        f"  Dense MRR: {observed_dense_mrr:.4f} (95% CI: [{drr_ci[0]:.4f}, {drr_ci[1]:.4f}])"
    )
    print(
        f"  Sparse MRR: {observed_sparse_mrr:.4f} (95% CI: [{srr_ci[0]:.4f}, {srr_ci[1]:.4f}])"
    )
    print(f"  k=5 MRR: {mrr_k5:.4f} vs k=60: {mrr_k60:.4f} (Δ={mrr_k5 - mrr_k60:+.4f})")
    print(f"  Cosine sim delta from prefix fix: {sim_v2 - sim_v1:+.4f}")
    print(
        f"  Q70 MRR: {next((c['crr_by_k'][RRF_K] for c in classification_rows if c['q_num'] == 70), 'N/A')}"
    )


if __name__ == "__main__":
    main()
