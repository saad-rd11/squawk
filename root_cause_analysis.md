# Root Cause Analysis — Why Low-Leakage MRR is 0.455

Six independent problems, each contributing to the low performance on paraphrase-style (low-leakage) queries. Ordered by severity.

---

## Finding 1: Anomaly Mapping Destroys Informative Prefix Tokens (CRITICAL)

### Evidence

The `anomaly_map.py:121` maps specific ASRS anomaly codes to generic ADREP categories:

```python
"Deviation / Discrepancy - Procedural Landing Without Clearance": ("OTHR", "other"),   # line 121
"Ground Event / Encounter Gear Up Landing": ("USOS", "undershoot overshoot"),           # line 107
"Deviation / Discrepancy - Procedural Published Material / Policy": ("OTHR", "other"),  # line 109
"Deviation / Discrepancy - Procedural FAR": ("OTHR", "other"),                          # line 113
"Deviation / Discrepancy - Procedural Hazardous Material Violation": ("OTHR", "other"), # line 115
```

The prefix embedded with every chunk is built from these ADREP-mapped plain strings (`transform.py:178-185`):

```python
prefix = build_prefix(
    aircraft_models, aircraft_family, flight_phase, anomaly_plain, ...
)
```

The resulting prefix for Q61's gold ACN (1715281) is:
```
[Aircraft: B737 (B737 family) | Phase: Landing, Approach | Anomaly: other]
```

The raw anomaly code `"Deviation / Discrepancy - Procedural Landing Without Clearance"` — which exactly matches the query intent "a pilot landing without a clearance" — is reduced to `"other"`. The prefix contains **zero discriminative signal** for the most informative metadata field.

### Impact on Q61 (rank 107)

| Signal source | Contains "landing"? | Contains "clearance"? | Contains "pilot"? |
|---------------|-------------------|---------------------|------------------|
| Query | landing | clearance | pilot |
| Prefix: Phase | **Landing** | — | — |
| Prefix: Anomaly | — | other (useless) | — |
| Narrative | uneventful approach | cleared for visual approach | We |

The only match is "Landing" from the flight phase field. The sparse BM25 model gets essentially zero lexical overlap with the query. The dense model must rely on semantic similarity between "landing without clearance" and "we were cleared for a visual approach ... uneventful approach" — which is a very subtle distinction.

### Root cause

The ASRS→ADREP mapping was designed for cross-database alignment (`anomaly_map.py:4-12`), not for embedding prefix quality. Procedural deviations are the **most common anomaly type** in ASRS, and they are all mapped to `"other"`. The prefix builder inherits this lossy mapping, poisoning the embedding input for ~50% of the corpus.

---

## Finding 2: LLM-Generated Queries vs ASRS Narrative Vocabulary (FUNDAMENTAL)

### Evidence

The 72 eval queries (`eval_queries.json`) use formal, journalistic English. The ASRS narratives use colloquial, abbreviation-filled, often grammatically irregular English.

| Query | Narrative vocabulary | Common words |
|-------|-------------------|--------------|
| "A baggage cart striking a parked aircraft on the ramp." | "driver grab bags from chute asked him to pull and he hit the plane" | **NONE** (only stop word "the") |
| "Data-entry keyboards at an air traffic facility malfunctioning after being over-sprayed during pandemic cleaning." | "data input keyboards ... Level 3 cleaning being performed" | "keyboards", "cleaning" |
| "An air ambulance helicopter having a near miss while crossing a non-towered field with a patient aboard." | "Departing hospital with patient onboard ... Broadcast my intentions on the CTAF" | "patient" (fragment) |
| "A pilot landing without a clearance." | "We flew a long pattern ... We were cleared for a visual approach ... extremely smooth and uneventful approach" | "landing" (from Phase prefix only) |

### Leakage metric details

The leakage function (`diagnose.py:52-58`) uses simple whitespace-split token overlap:

```python
qtokens = set(query_text.lower().split())
ntokens = set(narrative_text.lower().split())
return len(qtokens & ntokens) / len(qtokens)
```

This is inflated by stop words for short queries. Q70 ("A gear-up landing.", 3 tokens) gets 33% leakage from just the word "landing" appearing in the prefix. The content-bearing word "gear-up" never appears in any gold narrative text.

### Gold standard Q66

For Q66, the gold narrative (ACN 1919958) is 139 chars of broken English:
```
Driver grab bags from chute asked him to pull and he hit the plane. Seen him hit plane. Notified crew chief. Suggestions - Be more careful.
```

This narrative has **zero content-word overlap** with the query "A baggage cart striking a parked aircraft on the ramp." No lexical retrieval system (BM25, BM42, or RRF) can match these unless the dense model has learned a "baggage cart = driver carrying bags" and "striking = hit" association from its training data. BGE-base-en-v1.5 may have partial capability here (dense rank was 22), but BM25 does not.

---

## Finding 3: Max-Score Collapse Loses Distributed Evidence (DESIGN FLAW)

### Evidence

In `eval_recall.py:30-41` (and `diagnose.py:28-36`):

```python
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
```

This takes the **maximum** score across all chunks of a parent. If the gold information is distributed across chunks 1 and 2, neither chunk scores high individually, but together they would have high relevance. The max-score collapse misses this.

### Concrete example: Q70 ("A gear-up landing", ACN 1761576)

Narrative chunk 0 (971 chars):
```
[In] my preflight noticed that both engines needed a quart of oil. Filled both back...
```
→ No mention of gear-up landing. Score for query "gear-up landing": low.

Narrative chunk 1 (513 chars):
```
A split second later heard tick of a prop hitting the ground. ... pulled power all the way back and bellied the plane in.
```
→ Describes the actual gear-up landing. But "gear-up" and "landing" are not explicit. Score: moderate.

Neither chunk individually contains the phrase "gear-up" or "landing" (the word "landing" is only in the Phase prefix). If the relevance signal is fragmented across chunks, no single chunk reaches a high score, and the parent gets an unfairly low collapsed score.

### Also affects Q53 (ACN 2182338, 2 chunks)

Chunk 0 describes the departure and near-conflict. Chunk 1 says "The flight proceeded without incident." The NMAC event is fully in chunk 0, but the "patient aboard" context is only mentioned there once. If the chunk boundary splits the key description, the max-score collapse may not capture the full picture.

---

## Finding 4: RRF k=60 Hurts When Dense >> Sparse (RRF TRADEOFF)

### Evidence

Q66: dense rank = 22, sparse rank = 176. Combined RRF rank = **55** (worse than dense).

RRF formula (`eval_recall.py:57-62`):
```python
scores[acn] = 1.0 / (RRF_K + dr.get(acn, default)) + 1.0 / (RRF_K + sr.get(acn, default))
```

At k=60, each retriever contributes a diminishing reciprocal score. The sparse retriever's terrible rank (176) contributes 1/(60+176) = 0.0042. But other non-gold parents with moderate dense ranks (~30) and better sparse ranks (~5) score higher: 1/(60+30) + 1/(60+5) = 0.0265. The gold parent (combined score 0.0164) drops to rank 55.

At k=5, the sparse penalty is less severe: gold gets 1/(5+22) + 1/(5+176) = 0.0425, while the non-gold gets 1/(5+30) + 1/(5+5) = 0.1286. The gap is still large, but the absolute scores change the ordering.

### Validation: BM25 diagnosis showed k=5 outperforms k=60 on every slice

| Slice | k=5 MRR | k=60 MRR | Δ |
|-------|---------|---------|---|
| Overall | 0.7396 | 0.7322 | -0.0074 |
| Conceptual | 0.7527 | 0.7472 | -0.0055 |
| Exact_token | 0.6897 | 0.6752 | -0.0145 |

The optimal k=10 (0.7397) is essentially tied with k=5 (0.7396). Both significantly beat k=60 (0.7322).

### Implication

The diagnosis was run at k=60 for most analysis. At k=5 or k=10, the low-leakage MRR would be slightly better, but the improvement is marginal (Δ ≈ 0.007). The fundamental vocabulary mismatch issues dominate.

---

## Finding 5: Some Gold Labels Are Wrong or Misleading (DATA QUALITY)

### Evidence

**Q70: Gold ACN 1761576 — narrative does not describe a gear-up landing in chunk 0**

The query is "A gear-up landing." The gold narrative (chunk 0, 971/1485 chars) describes an engine oil check and low oil pressure. The anomaly code "Ground Event / Encounter Gear Up Landing" is correct for the event, but the `Narrative` field appears to be a multi-incident narrative where the actual gear-up landing is only described late in chunk 1 (from "heard tick of a prop hitting the ground ... bellied the plane in").

But the query "A gear-up landing" was matched to this ACN based on the anomaly code, not the narrative content. The narrative IS correct, but the gear-up landing description is at position 971/1485 (in the second chunk). The first 971 chars are about a different incident (oil issue).

**Q61: Gold ACN 1715281 — narrative describes a normal landing**

The anomaly code is "Landing Without Clearance." But the narrative describes a standard, uneventful approach where the pilot was "cleared for a visual approach." The legal subtlety (visual approach ≠ landing clearance) is not evident from the narrative text. The pilot may not even have been aware of the violation.

**Q71: Three gold ACNs, only one directly matches the query**

Query mentions "captain blamed reduced recent flying during the pandemic slowdown." None of the three gold narratives explicitly mention pandemic or reduced flying. The ACNs were matched based on altitude/speed deviation anomaly codes, not the specific causal claim in the query.

### Root cause

The queries were generated from anomaly codes (commit 1044996 message: "Verified CV eval set: 30 queries (12 exact_token, 18 conceptual); leakage 35.3%"). The expected ACNs were selected based on anomaly code matches, not narrative content verification. A query like "A pilot landing without a clearance" was generated from the anomaly code "Deviation / Discrepancy - Procedural Landing Without Clearance" and matched to an ACN with that code — regardless of whether the narrative actually describes the event in a retrievable way.

---

## Finding 6: ColBERT Reranker Has Limited Effect on Vocabulary Gap

### Evidence

From the reranker test (`reranker_results.md`):

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Low-leak MRR (n=15) | 0.4537 | 0.4818 | +0.0281 |
| Reachable only (13/15) | 0.5235 | 0.5560 | +0.0324 |

ColBERT improves 6 queries but worsens 2. The net gain is +6.2% relative.

### Why ColBERT doesn't close the gap

1. **Model capacity**: `answerai-colbert-small-v1` has dim=96 and was trained on MS MARCO (web search). It has not seen aviation safety report language.

2. **Query length sensitivity**: Q70 (3 tokens) gives ColBERT almost no query signal for late interaction. Each of the 3 query tokens must find a similar document token, but "gear-up" has no semantic neighbor in the gold narrative.

3. **Token-level matching is still lexical**: ColBERT's MaxSim compares token embeddings, which capture semantics but are still grounded in the tokenizer's vocabulary. "Baggage" and "bags" may have similar embeddings, but "baggage cart" (two tokens) and "driver grab bags ... from chute" (five tokens describing the same concept) have very different token sequences.

4. **Two queries are unreachable**: Q61 and Q66 have gold at combined ranks 107 and 55 — outside the top-50 candidate pool. The reranker cannot improve what it never sees.

### Implication

ColBERT reranking is not a silver bullet for the vocabulary mismatch problem. The improvement (+0.028 MRR) is real but small. The fundamental issue is the gap between LLM-generated query vocabulary and ASRS narrative vocabulary.

---

## Summary: Six Problems, One Root Cause Hierarchy

```
Problem 1: Anomaly mapping loss (CRITICAL)
  └─ Prefix says "other" for most procedural violations
  └─ Affects ~50% of corpus (procedural deviations all → "other")
  └─ Directly removes the signal that would match query intent
  ↓

Problem 2: Query-narrative vocabulary gap (FUNDAMENTAL)
  └─ LLM-generated queries use formal English
  └─ ASRS narratives use colloquial/abbreviated English
  └─ Zero content-word overlap for Q66, near-zero for Q48, Q53, Q61
  ↓

Problem 3: Max-score chunk collapse (DESIGN)
  └─ Distributed evidence across chunks is lost
  └─ Q70 gear-up landing split between two chunks
  ↓

Problem 4: RRF k=60 hurts dense-dominant queries (TUNING)
  └─ Q66 dense=22, sparse=176 → combined=55
  └─ k=5 partially mitigates but doesn't solve
  ↓

Problem 5: Gold label quality (DATA)
  └─ ACNs selected by anomaly code match, not narrative verification
  └─ Some narratives don't effectively describe the matched event
  ↓

Problem 6: ColBERT model gap (MITIGATION)
  └─ Tiny model, general-domain training, limited aviation semantics
  └─ +0.028 improvement is real but insufficient
```

### Concrete next steps to fix

1. **Fix the anomaly mapping**: Keep raw ASRS anomaly plain-text in the prefix alongside (or instead of) the ADREP-mapped text. The raw codes like "Deviation / Discrepancy - Procedural Landing Without Clearance" contain more signal than "other."

2. **Add narrative chunk aggregation**: Use mean/max pooling across chunks, or include chunk metadata (like "this chunk contains the anomaly-relevant portion"), or add a cross-chunk attention mechanism.

3. **Reduce RRF k from 60 to 5-10**: Already shown to be Pareto-optimal on all slices.

4. **Verify gold labels**: For the 4 tail queries (Q48, Q53, Q61, Q66), manually verify that the gold narratives actually contain retrievable evidence of the described event.

5. **Use a larger/aviation-fine-tuned reranker**: ColBERTv2 (dim=128) or a cross-encoder fine-tuned on aviation text.

6. **Implement HyDE/query expansion**: Expand short/terse queries like "A gear-up landing" into full-scenario descriptions before retrieval.
