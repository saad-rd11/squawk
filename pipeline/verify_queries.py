"""
Verification gates for eval_queries.json.

Run:
    python -m pipeline.verify_queries
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd


def _load_narratives(csv_path="pool_full_output.csv") -> dict[int, str]:
    """Load raw narratives keyed by ACN from the pool CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    narrs: dict[int, str] = {}
    for _, row in df.iterrows():
        acn = int(row["acn"])
        narr = row.get("Narrative", "")
        if pd.notna(narr):
            narrs[acn] = str(narr)
    return narrs


def gate1_acn_in_pool(queries: list[dict], pool: set[int]):
    """Drop expected_acns not in the pool. Drop queries left with zero ACNs."""
    print("=" * 60)
    print("GATE 1 — ACN-in-pool")
    print("=" * 60)
    dropped_acns = 0
    dropped_queries = []
    surviving = []
    for q in queries:
        good = [a for a in q["expected_acns"] if a in pool]
        bad = [a for a in q["expected_acns"] if a not in pool]
        for a in bad:
            print(f"  DROP ACN {a}  (outside pool)  — Q: {q['query'][:70]}")
            dropped_acns += 1
        if not good:
            dropped_queries.append(q)
            print(f"  DROP QUERY (no surviving ACNs) — Q: {q['query'][:70]}")
        else:
            q["expected_acns"] = good
            surviving.append(q)

    print(
        f"\n  Result: {dropped_acns} ACNs dropped, {len(dropped_queries)} queries dropped"
    )
    print(f"  Surviving: {len(surviving)} queries")
    return surviving


def gate2_anchor_grounding(queries: list[dict], narratives: dict[int, str]):
    """For exact_token queries, assert anchor string appears in narrative text."""
    print("\n" + "=" * 60)
    print("GATE 2 — Anchor grounding (exact_token only)")
    print("=" * 60)

    re_labels: list[tuple[str, str, list[int]]] = []

    for q in queries:
        if q.get("type") != "exact_token":
            continue
        anchors = q.get("anchor_tokens", [])
        if not anchors:
            print(f"  WARN: exact_token query has no anchor_tokens: {q['query'][:70]}")
            continue

        new_acns = []
        dropped = []
        for a in q["expected_acns"]:
            narr = narratives.get(a, "")
            # Check all anchors; if ANY anchor is missing, flag
            missing_anchors = []
            for tok in anchors:
                if tok.lower() not in narr.lower():
                    missing_anchors.append(tok)

            if missing_anchors:
                dropped.append((a, missing_anchors))
                print(
                    f"  FAIL anchor ({', '.join(missing_anchors)}) for ACN {a} "
                    f"in Q: {q['query'][:60]}"
                )
            else:
                new_acns.append(a)

        if dropped:
            # Check if ANY ACN grounded
            if new_acns:
                print(f"    → Kept ACNs {new_acns}, dropped {[d[0] for d in dropped]}")
            else:
                print(f"    → NO anchor grounded! Re-labeling to conceptual.")
                re_labels.append((q, "conceptual"))
            q["expected_acns"] = new_acns
        else:
            print(f"  OK ({len(new_acns)} ACNs grounded) — {q['query'][:60]}")

    # Re-label queries that lost all anchor grounding
    for q, new_type in re_labels:
        old_type = q.get("type", "unknown")
        print(f"\n  ** RE-LABEL: '{q['query'][:60]}' from {old_type} → {new_type}")
        q["type"] = new_type
        q["anchor_tokens"] = []

    # Summary
    exact = [q for q in queries if q.get("type") == "exact_token"]
    conc = [q for q in queries if q.get("type") == "conceptual"]
    print(f"\n  Result: {len(exact)} exact_token, {len(conc)} conceptual")
    return queries


def gate3_leakage_check(queries: list[dict], narratives: dict[int, str]):
    """Measure avg query-narrative token overlap for conceptual queries only."""
    print("\n" + "=" * 60)
    print("GATE 3 — Leakage re-measure (conceptual queries only)")
    print("=" * 60)

    overlap_scores = []
    for q in queries:
        if q.get("type") != "conceptual":
            continue
        expected = set(q["expected_acns"])
        q_tokens = set(re.findall(r"[a-z0-9]+", q["query"].lower()))

        expected_chunks = []
        for a in expected:
            narr = narratives.get(a, "")
            if narr:
                expected_chunks.append(narr)

        if not expected_chunks:
            continue

        all_narrative_text = " ".join(expected_chunks).lower()
        n_tokens = set(re.findall(r"[a-z0-9]+", all_narrative_text))

        if q_tokens:
            overlap = len(q_tokens & n_tokens) / len(q_tokens)
            overlap_scores.append(overlap)

    if overlap_scores:
        avg = sum(overlap_scores) / len(overlap_scores) * 100
        print(f"  Average query-narrative token overlap: {avg:.1f}%")
        if avg > 30:
            print(
                f"  *** FAIL: {avg:.1f}% > 30% threshold — query set needs re-draft ***"
            )
        else:
            print(f"  PASS: {avg:.1f}% ≤ 30% threshold")
        return avg
    else:
        print("  No conceptual queries to measure.")
        return 0.0


def gate4_report(queries: list[dict], original_count: int):
    """Report surviving set composition."""
    print("\n" + "=" * 60)
    print("GATE 4 — Surviving set report")
    print("=" * 60)

    exact = [q for q in queries if q.get("type") == "exact_token"]
    conc = [q for q in queries if q.get("type") == "conceptual"]
    total = len(queries)

    total_acns = sum(len(q["expected_acns"]) for q in queries)

    print(f"  Original queries: {original_count}")
    print(
        f"  Surviving queries: {total} ({len(exact)} exact_token, {len(conc)} conceptual)"
    )
    print(f"  Total expected_acns: {total_acns}")
    print(f"  Dropped: {original_count - total} queries")

    # Check for queries with type mismatch or other annotations
    for q in queries:
        if q.get("type") not in ("exact_token", "conceptual"):
            print(
                f"  WARN: query has unknown type '{q.get('type')}': {q['query'][:60]}"
            )

    return queries


def main():
    # Load queries
    with open("eval_queries.json") as f:
        queries = json.load(f)
    original_count = len(queries)
    print(f"Loaded {original_count} queries from eval_queries.json")

    # Load pool
    with open("cv_pool_acns.json") as f:
        pool = set(json.load(f))
    print(f"Pool: {len(pool)} ACNs")

    # Load narratives for gate 2 & 3
    print("Loading narratives from pool_full_output.csv...")
    narratives = _load_narratives()

    # === GATE 1 ===
    queries = gate1_acn_in_pool(queries, pool)

    # === GATE 2 ===
    queries = gate2_anchor_grounding(queries, narratives)

    # === GATE 3 ===
    leakage_pct = gate3_leakage_check(queries, narratives)

    # === GATE 4 ===
    queries = gate4_report(queries, original_count)

    # Write surviving set
    out_path = "eval_queries.json"
    with open(out_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"\nWrote surviving set to {out_path}")

    # Exit code for automation
    if leakage_pct > 30:
        print("\n*** GATE 3 FAILED — queries need re-draft ***")
        sys.exit(1)

    print("\n*** ALL GATES PASSED — ready for eval ***")


if __name__ == "__main__":
    main()
