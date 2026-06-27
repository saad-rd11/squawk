# ColBERT Reranker Test — Low-Leakage Conceptual Queries

Model: `answerdotai/answerai-colbert-small-v1` (dim=96)
Candidate pool: Combined (RRF k=60) top-50
Queries: 15 low-leakage conceptual (overlap ≤ 35%)

## Per-Query Results

| Q# | Leak% | C@1 | Col@1 | MRR before | MRR after | ΔMRR |
|----|-------|-----|-------|-----------|----------|------|
| Q17 | 35% | 1 | 1 | 1.0000 | 1.0000 | +0.0000 |
| Q24 | 21% | 1 | 1 | 1.0000 | 1.0000 | +0.0000 |
| Q37 | 31% | 2 | 1 | 0.5000 | 1.0000 | +0.5000 |
| Q38 | 31% | 5 | 4 | 0.2000 | 0.2500 | +0.0500 |
| Q42 | 25% | 1 | 1 | 1.0000 | 1.0000 | +0.0000 |
| Q44 | 30% | 1 | 1 | 1.0000 | 1.0000 | +0.0000 |
| Q48 | 26% | 25 | 13 | 0.0400 | 0.0769 | +0.0369 |
| Q51 | 25% | 12 | 11 | 0.0833 | 0.0909 | +0.0076 |
| Q52 | 20% | 5 | 3 | 0.2000 | 0.3333 | +0.1333 |
| Q53 | 33% | 31 | 12 | 0.0323 | 0.0833 | +0.0511 |
| Q60 | 14% | 1 | 1 | 1.0000 | 1.0000 | +0.0000 |
| Q61 | 20% | miss | miss | 0.0000 | 0.0000 | +0.0000 |
| Q66 | 11% | miss | miss | 0.0000 | 0.0000 | +0.0000 |
| Q70 | 33% | 2 | 4 | 0.5000 | 0.2500 | -0.2500 |
| Q71 | 35% | 4 | 7 | 0.2500 | 0.1429 | -0.1071 |

## Summary

- Combined MRR before: 0.4537
- ColBERT reranked MRR: 0.4818
- ΔMRR: +0.0281
- Improved: 6/15  Worsened: 2/15
- Gold in top-50 candidates: 13/15 (reachable MRR before=0.5235, after=0.5560)
