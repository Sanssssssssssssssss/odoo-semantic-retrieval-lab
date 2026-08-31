# Odoo Semantic Retrieval Lab

Independent, evidence-led laboratory for Odoo documentation retrieval and RAG evaluation.

P0-P4 are implemented and independently Agent-reviewed: deterministic Sphinx evidence extraction,
four chunkers, provisional Seed50 data, E0-E3 retrieval/reranking, heterogeneous pooling, metrics,
and a measured C2 performance baseline. Seed50 remains provisional until the human-review receipt
is completed; it is not a SOTA benchmark or frozen gold set. See [PLAN.md](PLAN.md) for the full
contract and the future SQL/PostgreSQL/pgvector boundary.

The benchmark-v2 contract now derives grades mechanically from topical relevance and required
atomic-nugget coverage. This invalidates the earlier depth-20/30/40/50 Agent pooling package;
that historical package remains fail-closed and is excluded until it is re-annotated and
adjudicated under the new contract. Formal `pooling_stable`, `human_review_complete`, and
`seed_frozen` remain false.

The delivered provisional corpus contains 50,350 EvidenceUnits and four deterministic chunk
variants (3,439 / 7,510 / 8,114 / 4,801 chunks). The rebuilt Seed50 contains 76 atomic nuggets,
145 canonical judgments, and 62 current seeded hard negatives. Statistical results remain
diagnostic only; none of them promote a retriever.

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
.\lab.ps1 tune-recall
.\lab.ps1 v0-bootstrap
.\lab.ps1 v0-validate --annotator annotator_a --submission <completed.jsonl>
.\lab.ps1 v0-adjudicate --annotator-a <a.jsonl> --annotator-b <b.jsonl>
.\lab.ps1 v0-validate-adjudication --root <private-root> --submission <adjudicator.jsonl> --annotator-a <a.jsonl> --annotator-b <b.jsonl>
.\lab.ps1 all -Profile seed
```

`tune-e2` runs the preregistered C2 CPU candidate screen: RRF/TMM fusion replay, proper
BM25F, deterministic parent/sibling EvidenceCard expansion, and PyTorch/ONNX/dynamic-int8
BGE query-encoder comparisons. `tune-e3` uses the isolated GPU environment to compare
cross-encoder candidate pools 10/20/50 and the preregistered sequence-length ablation. Both
commands write hash-bound local receipts under `artifacts/tuning/`; Seed50 remains provisional,
so a passing screen only earns evaluation on the future V0 benchmark.

`tune-recall` runs the approved C2/C3 recall-v2 screen: deterministic contextual metadata,
surface-only query decomposition, and source-ordinal parent/neighbor EvidenceCard expansion. It
uses the RTX 4070 only for offline corpus embedding and measures online queries on CPU float32.
See [reports/RECALL_V2.md](reports/RECALL_V2.md) for the approved provisional result.

`v0-bootstrap` starts that V0 work without creating Agent-authored gold. It freezes 160 empty
human-authored topic slots (96 public dev, 64 ignored shadow-hidden), canonicalizes the 9,185
chunk-level Seed review rows into 6,439 exact query/source-span groups, and emits a 20-item blind
tooling-calibration packet. The packet is Seed50 workflow validation only, not part of V0 gold.
Annotators see query text, frozen answerability, atomic nugget text, and candidate evidence; sampling
reasons, gold spans, and Agent labels remain hidden. `v0-validate` mechanically derives the grade and fail-closes
completed A/B submissions against the packet and evidence IDs. `v0-adjudicate` is the only way to
create adjudicator input; it first validates both mutually hidden A/B submissions and then writes a
disagreement-only packet under `.private/v0/` bound to both hashes.
Hidden topic content may only exist under the gitignored `.private/v0/` store; public files contain
only a count and hash commitment.

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
