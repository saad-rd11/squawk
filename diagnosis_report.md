# Stage 3 Hybrid Retrieval Diagnosis — Side-by-Side (BM42 vs BM25)

## Step 1: Corpus Baseline

| Property | Current | Previous (historical) |
|----------|---------|----------------------|
| `points.jsonl` total | 43,847 | same |
| Parents | 15,004 | same |
| Children | 28,843 | same |
| Pool ACNs | 250 | 250 |
| Pool children in Qdrant | 535 | 336 |
| `chunk_max_chars` at generation | 1024 | 2048 |

**Why 535 vs 336:** The original commit (`31ff0d8`) had `chunk_max_chars=2048`, producing ~336 children. Current working tree changed to 1024 (uncommitted), so the same 250 ACNs produce 535 finer chunks. Both runs use the same pool and the same `points.jsonl`.

**Current Qdrant collection:** 535 pool child points only. Distribution: 1 child (100 ACNs), 2 (84), 3 (35), 4 (17), 5+ (14).

---

## Step 2: Sparse Model Mismatch

| Property | Previous diagnosis run | This diagnosis run |
|----------|----------------------|-------------------|
| Config sparse_model | `Qdrant/bm25` | `Qdrant/bm25` |
| Qdrant doc sparse vectors | BM42 | BM25 (fresh index) |
| Query sparse vectors | BM25 (from config) | BM25 (from config) |
| Search type | **CROSS-MODEL** (BM25 q × BM42 doc) | **CONSISTENT** (BM25 q × BM25 doc) |

**Every sparse and combined RRF number in the previous report reflected a cross-model search.** The delta below isolates the document sparse model change. All other variables (pool, queries, dense model, chunk size) are held constant.

---

## Step 3: Re-index Action

```
stage3 --recreate --points points.jsonl --pool cv_pool_acns.json
```
Config: `sparse=Qdrant/bm25`, `dense=BAAI/bge-base-en-v1.5`, `chunk_max_chars=1024`. Only the sparse document vectors changed.

---

## Step 4: Side-by-Side Delta

### Baseline MRR

| Retriever | Slice | BM42 (cross-model) | BM25 (consistent) | Δ |
|-----------|-------|-------------------|-------------------|---|
| Dense | overall | 0.7229 | 0.7229 | 0 |
| Dense | conceptual | 0.7510 | 0.7510 | 0 |
| Dense | exact_token | 0.6160 | 0.6160 | 0 |
| Sparse | overall | 0.6893 | 0.6715 | -0.0178 |
| Sparse | conceptual | 0.6957 | 0.6833 | -0.0124 |
| Sparse | exact_token | 0.6650 | 0.6265 | -0.0385 |

### RRF — Full k Sweep

#### Overall MRR

| k | BM42 | BM25 | Δ |
|---|------|------|---|
| 5 | 0.7587 | 0.7396 | -0.0191 |
| 10 | 0.7549 | **0.7397** | -0.0152 |
| 20 | 0.7539 | 0.7361 | -0.0178 |
| 30 | 0.7538 | 0.7338 | -0.0200 |
| 40 | 0.7534 | 0.7331 | -0.0203 |
| 60 | 0.7534 | 0.7322 | -0.0212 |
| 80 | 0.7531 | 0.7321 | -0.0210 |

#### Conceptual MRR

| k | BM42 | BM25 | Δ |
|---|------|------|---|
| 5 | **0.7775** | **0.7527** | -0.0248 |
| 10 | 0.7749 | 0.7527 | -0.0222 |
| 20 | 0.7734 | 0.7522 | -0.0212 |
| 30 | 0.7732 | 0.7492 | -0.0240 |
| 40 | 0.7741 | 0.7484 | -0.0257 |
| 60 | 0.7742 | 0.7472 | -0.0270 |
| 80 | 0.7739 | 0.7470 | -0.0269 |

#### Exact_token MRR

| k | BM42 | BM25 | Δ |
|---|------|------|---|
| 5 | 0.6871 | 0.6897 | +0.0026 |
| 10 | 0.6790 | **0.6901** | +0.0111 |
| 20 | 0.6800 | 0.6750 | -0.0050 |
| 30 | 0.6804 | 0.6751 | -0.0053 |
| 40 | 0.6748 | 0.6749 | +0.0001 |
| 60 | 0.6743 | 0.6752 | +0.0009 |
| 80 | 0.6743 | 0.6752 | +0.0009 |

#### Best k per slice

| Slice | BM42 best k (MRR) | BM25 best k (MRR) |
|-------|--------------------|--------------------|
| Overall | 5 (0.7587) | 10 (0.7397) |
| Conceptual | 5 (0.7775) | 5 (0.7527) |
| Exact_token | 5 (0.6871) | 10 (0.6901) |

### Does combined beat dense on BM25?

| Scenario | Combined best | Dense alone | Δ | Beats dense? |
|----------|--------------|-------------|---|-------------|
| BM42 overall | 0.7587 | 0.7229 | +0.0358 | YES |
| **BM25 overall** | **0.7397** | **0.7229** | **+0.0168** | **YES** |
| BM42 conceptual | 0.7775 | 0.7510 | +0.0265 | YES |
| BM25 conceptual | 0.7527 | 0.7510 | +0.0017 | YES (marginal) |
| BM42 exact_token | 0.6871 | 0.6160 | +0.0711 | YES |
| BM25 exact_token | 0.6901 | 0.6160 | +0.0741 | YES |

Combined still beats dense on BM25 (0.7397 > 0.7229). The hybrid benefit survives, but the margin narrows by half (from +0.036 to +0.017). On conceptual, the margin is essentially zero (+0.002). On exact_token, the margin is strong (+0.074) and slightly wider than BM42.

### Does sparse conceptual MRR collapse toward ~0.27?

Sparse conceptual: 0.696 (BM42) → 0.683 (BM25). Δ = -0.012. **No collapse.** The drop is minimal. The ~0.27 historical value was on a different corpus/pipeline state.

### Regressions

| Metric | BM42 | BM25 | Δ | Status |
|--------|------|------|---|--------|
| Exact_token combined(k=60) vs sparse | +0.009 | +0.049 | +0.040 | Fixed |
| Multi-ACN@5 combined(k=60) vs dense | -0.018 | +0.030 | +0.048 | **Fixed** |

Both regressions observed with the BM42 mismatch disappear with consistent BM25. The multi-ACN regression (combined losing to dense) was a BM42 artifact.

### Reranker Ceiling (k=60, cutoff=50)

| | BM42 | BM25 |
|---|------|------|
| Gold in candidates (cutoff=50) | 70/72 (97%) | 70/72 (97%) |
| First-stage recall failure | 2/72 (3%) | 2/72 (3%) |
| In candidates but rank >20 | 2/72 (3%) | 2/72 (3%) |
| Gold in candidates (cutoff=100) | 72/72 (100%) | 71/72 (98.6%) |

### Per-Type Weighting

| Check | BM42 | BM25 |
|-------|------|------|
| Configs beating k=60 both slices | 0 | 2 |
| Configs beating k=5 both slices | 0 | 0 |

Two BM25 configs beat k=60 overall: `w_sparse_exac=0.5 w_dense_conc=0.7` (overall=0.7433) and `w_sparse_exac=0.5 w_dense_conc=0.9` (overall=0.7423). Neither beats k=5.

---

## Reranker Test: ColBERT Late Interaction (answerai-colbert-small-v1)

Candidates: Combined (RRF k=60) top-50 parent narratives. Queries: 15 low-leakage conceptual. Model dim=96.

| Q# | Leak% | C@1 | Col@1 | MRR before | MRR after | ΔMRR |
|----|-------|-----|-------|-----------|----------|------|
| Q17 | 35% | 1 | 1 | 1.0000 | 1.0000 | 0 |
| Q24 | 21% | 1 | 1 | 1.0000 | 1.0000 | 0 |
| Q37 | 31% | 2 | 1 | 0.5000 | 1.0000 | +0.5000 |
| Q38 | 31% | 5 | 4 | 0.2000 | 0.2500 | +0.0500 |
| Q42 | 25% | 1 | 1 | 1.0000 | 1.0000 | 0 |
| Q44 | 30% | 1 | 1 | 1.0000 | 1.0000 | 0 |
| Q48 | 26% | 25 | 13 | 0.0400 | 0.0769 | +0.0369 |
| Q51 | 25% | 12 | 11 | 0.0833 | 0.0909 | +0.0076 |
| Q52 | 20% | 5 | 3 | 0.2000 | 0.3333 | +0.1333 |
| Q53 | 33% | 31 | 12 | 0.0323 | 0.0833 | +0.0511 |
| Q60 | 14% | 1 | 1 | 1.0000 | 1.0000 | 0 |
| Q61 | 20% | miss | miss | 0.0000 | 0.0000 | 0 |
| Q66 | 11% | miss | miss | 0.0000 | 0.0000 | 0 |
| Q70 | 33% | 2 | 4 | 0.5000 | 0.2500 | -0.2500 |
| Q71 | 35% | 4 | 7 | 0.2500 | 0.1429 | -0.1071 |

**Summary:** Combined MRR: 0.4537 → ColBERT: **0.4818** (Δ = +0.0281, +6.2% relative). Improved: 6/15. Worsened: 2/15. Tied: 7/15.

ColBERT helps modestly on low-leakage conceptual queries. Gains are real: Q37 (2→1), Q53 (31→12), Q48 (25→13), Q52 (5→3). Two queries are unreachable (Q61, Q66 — first-stage recall failure). Two regress slightly (Q70 2→4, Q71 4→7). The post-rerank low-leak MRR (0.482) is still far below high-leakage (0.851) — late interaction does not close the paraphrase-vocabulary gap.

---

## Step 5: Leakage-Stratified Results

**Threshold:** Low-leak = token overlap ≤ 35.3% (corpus avg). High-leak = above 35.3%.

**Corpus-wide leakage average:** 46.4%

### Conceptual MRR by leakage tier (k=60)

| Metric | Low-leak (n=15) | High-leak (n=42) | Corpus (n=57) | Gap (low vs corpus) |
|--------|----------------|-----------------|---------------|-------------------|
| Combined | **0.4555** | 0.8514 | 0.7472 | -39.0% |
| Dense | 0.4471 | 0.8596 | 0.7510 | -40.5% |
| Sparse | 0.4129 | 0.7799 | 0.6833 | -39.6% |

**Low-leakage combined (0.4555) barely beats dense (0.4471)** — hybrid offers negligible benefit on genuinely paraphrase queries.

**BM25 vs BM42 on low-leakage only:**

| | BM42 low-leak | BM25 low-leak | Δ |
|---|--------------|--------------|---|
| Combined | 0.5174 | 0.4555 | -0.0619 |
| Sparse | 0.4468 | 0.4129 | -0.0339 |
| Dense | 0.4471 | 0.4471 | 0 |

BM25 combined is worse than BM42 on low-leakage queries (0.4555 vs 0.5174). The dense number is identical.

### Full BM25 metrics by leakage tier

#### All 72 queries
- Combined (k=60): low-leak conc=0.4555 (n=15), high-leak conc=0.8514 (n=42), exact_token=0.6752 (n=15)
- Dense: low-leak conc=0.4471 (n=15), high-leak conc=0.8596 (n=42), exact_token=0.6160 (n=15)
- Sparse: low-leak conc=0.4129 (n=15), high-leak conc=0.7799 (n=42), exact_token=0.6265 (n=15)

#### Multi-ACN recall (non-tail, n=14)
| @k | BM42 | BM25 | Δ |
|----|------|------|---|
| @1 | 0.292 | 0.292 | 0 |
| @5 | 0.601 | 0.649 | +0.048 |
| @10 | 0.780 | 0.827 | +0.047 |
| @20 | 0.905 | 0.952 | +0.047 |
| @50 | 0.952 | 0.952 | 0 |

#### Tail queries (Combined rank > 20, n=4, all conceptual)

| Q# | D@1 | S@1 | C@1 | Leak% |
|----|-----|-----|-----|-------|
| Q48 | 57 | 12 | 25 | 26 |
| Q53 | 25 | 50 | 31 | 33 |
| Q66 | 22 | 176 | 55 | 11 |
| Q61 | 82 | 111 | 107 | 20 |

Tail unchanged in composition vs BM42. Avg leakage: tail = 22.7%, non-tail = 47.8%.

### Distribution Shape (BM25, Combined k=60)

| Bucket | BM42 (%) | BM25 (%) | Δ |
|--------|---------|---------|---|
| Rank 1 | 62.5% | 58.3% | -4.2 |
| Ranks 2–5 | 16.7% | 25.0% | +8.3 |
| Ranks 6–20 | 15.3% | 11.1% | -4.2 |
| Tail (21+) | 5.6% | 5.6% | 0 |

BM25 spreads the distribution: fewer @1 hits but more @2-5 hits. Tail unchanged.
