# Odoo Semantic Retrieval Lab

Independent, evidence-led laboratory for Odoo documentation retrieval and RAG evaluation.

P0-P4 are implemented and independently Agent-reviewed: deterministic Sphinx evidence extraction,
four chunkers, provisional Seed50 data, E0-E3 retrieval/reranking, heterogeneous pooling, metrics,
and a measured C2 performance baseline. Seed50 remains provisional until the human-review receipt
is completed; it is not a SOTA benchmark or frozen gold set. See [PLAN.md](PLAN.md) for the full
contract and the future SQL/PostgreSQL/pgvector boundary.

The depth-20/30/40/50 pooling diagnostic has reached its preregistered saturation rule, but this is
recorded only as `agent_diagnostic_pooling_stable=true`; formal `pooling_stable`,
`human_review_complete`, and `seed_frozen` remain false.

The delivered provisional corpus contains 50,350 EvidenceUnits and four deterministic chunk
variants (3,439 / 7,510 / 8,114 / 4,801 chunks). The depth-50 pool contains 16,738 judged
candidates, 120 selected hard negatives, and a 9,185-row human-review queue. Statistical results
remain diagnostic only; none of them promote a retriever.

## Run

```powershell
.\lab.ps1 verify
.\lab.ps1 extract
.\lab.ps1 chunk
.\lab.ps1 smoke
.\lab.ps1 baseline
.\lab.ps1 pool
.\lab.ps1 perf
.\lab.ps1 p5
.\lab.ps1 tune-e2
.\lab.ps1 tune-e3
.\lab.ps1 all -Profile seed
```

`tune-e2` runs the preregistered C2 CPU candidate screen: RRF/TMM fusion replay, proper
BM25F, deterministic parent/sibling EvidenceCard expansion, and PyTorch/ONNX/dynamic-int8
BGE query-encoder comparisons. `tune-e3` uses the isolated GPU environment to compare
cross-encoder candidate pools 10/20/50 and the preregistered sequence-length ablation. Both
commands write hash-bound local receipts under `artifacts/tuning/`; Seed50 remains provisional,
so a passing screen only earns evaluation on the future V0 benchmark.

Every cross-stage command is receipt-gated. `all -Profile seed` checks every Agent approval before
its first write and returns an explicit `agent_provisional_complete_human_review_pending` state when
the human receipt is absent. It never freezes Seed50 or enables SOTA claims by inference.

Human review starts from
`benchmarks/seed50/pooling/provisional/human_review/decisions.template.jsonl`; a real reviewer must
complete it and issue `receipt.json`. The repository intentionally contains only a pending template.

Canonical, versioned inputs live under `corpus/`, `benchmarks/`, `configs/`, `schemas/`, and
`reviews/`. Rebuildable runs, matrices, indexes, models, and performance receipts are ignored local
artifacts. The only CLI entrypoint is `lab.ps1`.

## Isolation

The lab has its own Python environment, dependency lock, caches, corpus, indexes, and write
allowlist. It has no import, runtime, or write dependency on `erp-openai`, `erp-agent-odoo`, a live
Odoo instance, or the main Agent project.

The downloaded source checkout under `corpus/raw/` is intentionally ignored. Its versioned snapshot
receipt remains trackable and is sufficient to reacquire and verify the corpus.
