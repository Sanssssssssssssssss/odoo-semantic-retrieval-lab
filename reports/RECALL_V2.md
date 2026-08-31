# Recall v2 provisional experiment

Final approved run: `720af934f03e2587e2aee4f545ec65f8030d58a3`.

This is a Seed50 candidate screen, not formal gold, a production promotion, or a SOTA claim. Fixed-chunker metrics are only interpreted within C2 or C3. C2 versus C3 comparisons use the 2,048-token evidence measures.

## Fixed-chunker quality

| Chunk | Variant | nDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Hard-negative exposure |
|---|---|---:|---:|---:|---:|---:|---:|
| C2 | M0 baseline E2 | 0.7956 | 0.5625 | 0.7604 | 0.8875 | 0.9375 | 0.70 |
| C2 | M1 contextual E2 | 0.8048 | 0.5813 | 0.7625 | 0.9125 | 0.9425 | 0.75 |
| C2 | M1 + M2 surface decomposition | 0.7785 | 0.5687 | 0.7292 | 0.8812 | 0.9175 | 0.73 |
| C3 | M0 baseline E2 | 0.8203 | 0.5729 | 0.8125 | 0.9062 | 0.9563 | 0.75 |
| C3 | M1 contextual E2 | 0.8337 | 0.5979 | 0.8625 | 0.9062 | 0.9500 | 0.79 |
| C3 | M1 + M2 surface decomposition | 0.8188 | 0.5979 | 0.8313 | 0.8875 | 0.9437 | 0.76 |

M1 adds only deterministic Odoo version, module path, canonical page path, and ordered node types. M2 is a surface-text-only diagnostic that keeps the original query, creates at most four obvious action clauses, and performs a second RRF with the original query weighted 1.0 and all subqueries weighted 1.0 in total.

## Evidence expansion at 2,048 tokens

| Chunk | Mode | Evidence recall | Required-nugget recall | Completeness | Irrelevant-token ratio | Duplicate rate |
|---|---|---:|---:|---:|---:|---:|
| C2 | Leaf | 0.8567 | 0.8438 | 0.8000 | 0.9322 | 0.0000 |
| C2 | Neighbor | 0.8775 | 0.8750 | 0.8500 | 0.9592 | 0.0583 |
| C2 | Parent + neighbor | 0.8900 | 0.8875 | 0.8750 | 0.9629 | 0.0914 |
| C3 | Leaf | 0.8950 | 0.8875 | 0.8500 | 0.9506 | 0.0000 |
| C3 | Neighbor | 0.9125 | 0.9125 | 0.9000 | 0.9618 | 0.0490 |
| C3 | Parent + neighbor | 0.9000 | 0.9000 | 0.9000 | 0.9639 | 0.1071 |

Neighbor expansion now follows EvidenceUnit source ordinal, not chunk-ID order. C3 neighbor is the strongest evidence-coverage candidate; parent + neighbor adds duplication without improving completeness over neighbor.

## CPU warm latency

The RTX 4070 was used only for offline contextual corpus embedding. Online query execution was CPU float32, single-thread, warm, with 1,024 requests per row.

| Chunk | Variant | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|
| C2 | M1 contextual E2 | 32.074 | 38.369 | 44.032 |
| C2 | M1 + M2 | 33.625 | 93.290 | 99.974 |
| C3 | M1 contextual E2 | 32.434 | 42.122 | 48.131 |
| C3 | M1 + M2 | 31.814 | 88.609 | 97.954 |

Offline float32 dense-index builds took 50.17 seconds for C2 (12,463,104 bytes) and 29.28 seconds for C3 (7,374,336 bytes).

## Decision

- Keep the current E2 default unchanged. None of the four M1/M2 candidates passes every preregistered promotion guardrail.
- Carry `C2 + M1 contextual E2` as the recall-balanced research line: Recall@5 rises 2.5 percentage points, but hard-negative exposure rises from 0.70 to 0.75 and no-answer AUROC falls from 0.9125 to 0.855.
- Carry `C3 + M1 contextual E2 + neighbor EvidenceCard expansion` as the quality research line: it has the strongest nDCG@10 and evidence/nugget coverage, but Recall@10 falls 0.625 percentage points and hard-negative exposure rises from 0.75 to 0.79.
- Reject M2 as currently implemented. It reduces Recall@5/10 and creates large tail latency.

The approved receipt is `reviews/benchmark-recall-v2/approval.json`. Full local run artifacts remain ignored under `artifacts/tuning/recall-v2/`.
