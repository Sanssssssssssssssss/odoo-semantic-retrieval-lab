# Odoo Semantic Retrieval Lab — Evidence-led implementation plan

Status: P0-P4 implemented and independently Agent-reviewed. Seed50 and pooled judgments remain
provisional pending the required human review; no frozen-gold, SOTA, or main-agent claim is allowed.

## 1. Product contract

The lab is a standalone information-retrieval and RAG evaluation system for Odoo knowledge. It is
not a conversational RAG application.

The first useful outcome is a reproducible comparison of retrieval pipelines that, given an Odoo
semantic need, returns zero to four short, version-correct, authority-correct, source-grounded
evidence cards with measured accuracy, latency, and resource cost.

The lab must remain independent from `erp-openai` and `erp-agent-odoo`:

- separate repository, environment, caches, corpus, indexes, run artifacts, and secrets;
- no imports from, writes to, or runtime dependency on either main project;
- no live Odoo, MCP, Agent, Compiler, Verifier, or benchmark integration in V0;
- future Agent traces enter only as redacted, hashed, one-way dataset snapshots;
- mainline integration is considered only after a frozen retrieval benchmark passes independent
  review, and then only through a narrow read-only retrieval contract.

## 2. Acquired source snapshot

The official Odoo documentation is checked out at:

`corpus/raw/odoo-documentation-19.0`

The immutable receipt is:

`corpus/raw/odoo-documentation-19.0.snapshot.json`

Current snapshot:

- repository: `https://github.com/odoo/documentation.git`;
- source ref: `19.0`;
- commit: `32a8b8d77833f22b4bc74ed4ea78b6a82b5338fd`;
- commit time: `2026-08-29T09:35:27Z`;
- license: CC-BY-SA-4.0;
- sparse paths: `content`, `extensions`, `invs`, `redirects`, `static`, `tests`;
- intentionally excluded for V0: `locale` translations;
- checkout size excluding `.git`: 5,104 files / 129,139,359 bytes;
- English RST corpus: 1,156 files.
- snapshot manifest SHA-256:
  `C779918DAA43C93EF715C3A83CE759019A82629F23FB22C987FC1B0EC599DFAD`.

The checkout is clean and detached at the pinned commit. A branch name is never sufficient
provenance; every corpus build must bind to the commit and manifest hash.

## 3. What the real documentation implies

The corpus is Sphinx/reStructuredText, not Markdown or plain HTML. Inspection found:

- 935 application RST files and 147 developer RST files;
- 9,921 heading-defined semantic sections in a read-only structural audit;
- section character distribution: p50 778, p75 1,327, p90 2,084, p95 2,686,
  p99 4,458, maximum 57,441;
- 2,566 images, 1,915 notes, 1,098 see-also blocks, 1,004 tips, 760 important blocks;
- 723 code blocks, 251 list tables, 196 toctrees, 93 tabs, and 78 includes;
- custom roles and directives, anchors, substitutions such as `|PO|`, cross-document references,
  menu selections, GUI labels, tables, code, images with alt text, and optional autodoc content.

Therefore a production chunker must not be a regular-expression heading splitter. It would leave
unexpanded substitutions, lose toctree breadcrumbs, separate table rows from headers, flatten tabs,
drop include content, and make source citations unreliable.

Odoo already ships its Sphinx extensions and heading validation rules. The first extractor should
reuse the official Sphinx build/doctree rather than reimplement RST parsing.

## 4. Canonical pipeline

```text
Pinned SourceSnapshot
        |
        v
Official Sphinx parse/doctree
        |
        v
Canonical EvidenceUnits + source spans
        |
        +-----------------------+
        |                       |
        v                       v
CorpusVariant A/B/C       Benchmark qrels/rubrics
(chunker config)          (evidence-unit level)
        |                       |
        v                       |
Retriever / reranker ----------+
        |
        v
TREC run + metrics + stage timings + resource evidence
```

The critical separation is `EvidenceUnit != Chunk`.

An evidence unit is a stable, source-grounded block produced from the Sphinx document tree. A chunk
is an experiment-specific grouping of evidence units. Relevance judgments target evidence units,
so changing chunk size, overlap, or parent-child grouping does not invalidate the benchmark.

## 5. Evidence extraction contract

The Sphinx extraction spike must emit ordered evidence units for:

- section titles and heading paths;
- paragraphs and list items;
- admonitions with their type;
- tables with headers and rows preserved;
- literal/code blocks with language metadata;
- image alt text, but not image pixels in V0;
- resolved substitutions, internal references, document links, and anchors.

Each evidence unit carries at minimum:

```text
evidence_unit_id
source_snapshot_id
docname / source_path / public_source_uri
anchor_ids
heading_path
node_type
ordinal
source_start_line / source_end_line when available
rendered_text
lexical_text
content_hash
authority / odoo_version / corpus_scope
```

`rendered_text` is human-readable evidence. `lexical_text` may retain exact Odoo identifiers and UI
labels for BM25. Both remain derivations of the same source block and must be reproducible.

Extraction acceptance is based on parse/build evidence, not a guessed percentage: all in-scope
documents must either produce units or appear in an explicit exclusion report with a reason. All
chunks must map back to one or more evidence units and a source URI.

Formal benchmark annotation is blocked until this schema, text-normalization version, and stable
source-span mapping are frozen. An unresolved include, substitution, cross-reference, or autodoc
node is an explicit extraction error or declared exclusion; it must never silently become benchmark
text.

## 6. Corpus scopes

Corpus families are separate benchmark dimensions, not silently mixed data:

1. `applications`: primary business-semantic corpus and first benchmark scope.
2. `developer`: technical reference/tutorial corpus, added as a separately reported slice.
3. `administration`: operational knowledge, added only for corresponding queries.
4. `community_source` and `community_tests`: later Odoo 19 Community source snapshots with their
   own license and authority class.
5. live schema: never part of the static corpus; it is a separate runtime authority.
6. forum, blogs, third-party modules, Enterprise/private sources: excluded from V0.

The first meaningful retrieval benchmark uses the full `applications` text corpus rather than only
the documents that generated the questions, so near-duplicate and cross-module negatives remain
realistic.

## 7. Chunk experiments

Chunking is a measured experimental variable. V0 should build these corpus variants from the same
evidence-unit snapshot:

- `C0-fixed`: fixed-token control, including a published-style 512-token / 100-token-overlap
  condition for comparison with existing RAG retrieval work;
- `C1-section`: one Sphinx semantic section per chunk, with no routine overlap;
- `C2-structure-bounded`: semantic sections with soft/hard token limits; only oversized units are
  split, at sentence/list-row boundaries where possible;
- `C3-structure-merged`: C2 plus deterministic merging of undersized adjacent sibling sections;
- parent-child expansion is deferred until the above establish whether it is needed.

The observed section distribution supports testing multiple hard caps rather than choosing one by
intuition. Candidate caps are a predeclared grid and the winner is selected on the development set
by the quality/latency/context Pareto frontier, then evaluated once on the frozen test set.

Rules common to all structure-aware variants:

- preserve heading path and parent identity as metadata;
- keep tables isolated with repeated headers when a large table must be split;
- keep admonition type and its complete body together;
- keep code blocks atomic unless they exceed the hard model limit;
- avoid overlap across clean semantic boundaries; overlap is permitted only for forced splits and
  is measured as duplication cost;
- store the ordered evidence-unit IDs covered by every chunk;
- derive chunk IDs from source snapshot, chunker config hash, and ordered evidence-unit IDs.

The design follows mature behavior visible in Unstructured's `chunk_by_title` (semantic title
boundaries, soft/hard caps, isolated tables, cautious overlap) and Haystack's `DocumentSplitter`
(explicit length/overlap/threshold, source lineage metadata), without taking either framework as a
runtime dependency before the Odoo-specific extraction spike proves it useful.

## 8. Benchmark architecture

The benchmark, provisionally OSRB, has distinct tasks.

Its portable release package is the contract, independent of any database implementation:

```text
manifest.json
source_documents.jsonl
source_spans.jsonl
queries.jsonl
nuggets.jsonl
qrels.<split>.trec
splits.json
hard_negatives.jsonl
chunk_configs.jsonl
runs/<run_id>.trec
```

Every record is bound to the benchmark version, source snapshot, and content hash. Rich annotation
rationale and provenance live beside these exchange formats, while the TREC/BEIR-compatible files
remain sufficient to run standard retrieval evaluation.

### R — Retrieval

Input: query plus an optional structured Odoo context. Output: a ranked list of chunk IDs.

This task uses BEIR-compatible `corpus.jsonl`, `queries.jsonl`, and split qrels, and emits a standard
six-column TREC run file in addition to richer JSONL traces. Standard formats make external
`trec_eval`, BEIR, ranx, or MTEB-style evaluators possible without an adapter to internal objects.

### R1 — Reranking

Input: a frozen candidate pool. Output: a reranked candidate list. Fixing the candidate set follows
the MTEB reranking model and prevents first-stage recall changes from being credited to the reranker.
Every report includes the candidate-pool recall ceiling and the reranker's incremental latency.

### AG — Augmented generation, later

Input: query plus a fixed retrieved evidence set. This isolates generator quality from retrieval.

### RAG — End to end, later

Input: query and corpus. Output: answer with evidence citations. This is not enabled until the R
task is stable.

This separation follows the TREC RAG track's Retrieval, Augmented Generation, and end-to-end RAG
tasks. TREC RAG's requirement that custom chunks map back to canonical collection segments directly
supports the evidence-unit mapping above.

## 9. Query and judgment design

Queries are stratified by information need rather than generated as one undifferentiated QA set:

- model/field semantics;
- workflow prerequisite;
- cross-module relationship;
- action semantics and side effects;
- diagnostic/error recovery;
- exact identifier/code lookup;
- ambiguous near-neighbor and version conflict;
- unanswerable/out-of-scope queries for abstention.

Each benchmark query records its origin class: official-document reverse question, ERP task,
redacted Agent trace, or expert-authored adversarial query. Generated questions are drafts, never
automatic ground truth.

Judgments are graded and evidence-based:

```text
0 = not relevant / contradicts the requested version or authority
1 = topically related but insufficient
2 = useful supporting evidence
3 = directly necessary evidence
```

For multi-part needs, a query also owns required evidence rubrics/nuggets. Each nugget maps to one
or more evidence units. This lets evaluation distinguish complete evidence coverage, partial hit,
and total miss instead of treating any related chunk as success.

Qrels are built with TREC-style pooling across deliberately different systems: BM25, dense,
hybrid, and reranked runs, plus expert-found evidence. Unjudged candidates are not casually treated
as proven negatives in error analysis. Important judgments require a second review and adjudication;
the benchmark stores annotator, rationale, evidence span, and adjudication status.

The initial pool includes top results from structurally different retrieval families, seeded gold
evidence, sibling headings, lexical/UI collisions, adjacent modules, wrong versions, and observed
baseline false positives. Pool depth is extended until the new-relevant yield and leave-one-run-out
ranking are stable; it is not frozen at an arbitrary depth. Until that saturation evidence exists,
reports include `judged@k` plus bpref/condensed metrics and do not present ordinary nDCG/MAP as if
the qrels were complete. A genuinely new retrieval family triggers a pooling audit.

Hard negatives must be judged grade 0 or 1 and carry a provenance class such as sibling section,
lexical collision, semantic nearest neighbor, wrong module, wrong version, procedure-step
confusion, or baseline false positive. Hidden-test negatives never enter training or tuning.

Chunk-level qrels are derived from evidence-unit coverage for each corpus variant. Human labels do
not point directly at a single experimental chunk layout.

Within one frozen chunker, standard chunk-level nDCG/MAP comparisons are valid. Across different
chunkers, raw chunk-ID qrels are not directly comparable; chunks are projected back to canonical
evidence units and compared using evidence/nugget recall at a fixed context-token budget,
required-nugget completeness, irrelevant-token ratio, source-document nDCG, and duplicate-evidence
rate. Derived chunk qrels are caches keyed by `chunk_config_hash`, never benchmark gold.

## 10. Splits and anti-overfitting

The benchmark grows in gates:

- `seed`: enough reviewed queries to validate data contracts and expose labeling problems; no
  algorithm-quality claim is allowed;
- `dev`: public queries/qrels for iteration and failure analysis;
- `test`: frozen queries/qrels, used only for declared candidate comparisons;
- `hidden`: retained queries or qrels used for release/promotion decisions once repeated tuning
  begins.

V0 is a 160-topic adjudicated pilot: 96 development topics and 64 shadow-hidden topics, grouped by
source/fact clusters rather than randomly. It is large enough to exercise every taxonomy and
no-answer reason, but it is not a claim of SOTA. V1 uses `max(500, N)` topics, where `N` is computed
from V0 paired-score variance for a preregistered minimum practical improvement. A round-number
sample size alone is not treated as scientific justification.

Each topic is double-annotated and all disagreements are adjudicated. The release reports raw
agreement plus weighted kappa or alpha. LLMs may suggest candidates but cannot produce final gold.
Stratification must maintain coverage of query type, module, difficulty, answerability, version
conflict, evidence topology, and source family in every evaluable split.

Leakage controls:

- source pages, parent workflows, nugget/fact signatures, connected multi-hop evidence, and query
  paraphrases form a grouping graph; an entire connected component stays in one split;
- a generated query cannot be both tuning material and hidden evaluation;
- model or reranker training examples are fingerprinted against benchmark queries/evidence;
- multiple paraphrases of one semantic need stay in the same split;
- chunker parameters are selected only on dev;
- the test run declaration fixes corpus, code, model revisions, config, and primary metric before
  execution;
- benchmark and corpus manifests are content-addressed.

No-answer cases carry a declared reason: `absent_from_corpus`, `requires_live_instance`,
`wrong_version`, `enterprise_or_excluded`, `ambiguous_requires_clarification`, or `out_of_scope`.
The decision task is `retrieve / abstain / clarify`; thresholds are calibrated on dev and frozen
before hidden evaluation. Report macro-F1, abstention precision, coverage, and a risk-coverage curve.

## 11. Metrics

### Standard retrieval metrics

- nDCG@10 as the standard graded-ranking headline for external comparability;
- nDCG@3 and Precision@3 for the operational top-three evidence budget;
- Recall@3 and Recall@20;
- MRR@10;
- MAP where binary/graded judgment coverage makes it meaningful.

### Evidence and Odoo guardrails

- required-nugget/evidence coverage at k;
- complete / partial / miss rate;
- evidence/nugget recall at a fixed context-token budget;
- relevant-evidence tokens divided by total returned tokens;
- duplicate context ratio caused by overlap;
- authority violation rate;
- Odoo-version violation rate;
- abstention precision, recall, and coverage;
- source-citation validity.

RAGAS context precision/recall and RAGBench/TRACe concepts are useful secondary views, but no
LLM-as-judge score becomes the primary retrieval ground truth. Later end-to-end evaluation adds
TREC-RAG-style nugget completeness, citation support/correctness, and answer grounding.

### Latency and resource protocol

Measure and report separately:

- source extraction and index build time;
- model/index cold load time;
- warm query encoding, lexical search, vector search, fusion, reranking, and total latency;
- p50, p90, p95, and p99 over a deterministic cycle of real Seed queries;
- completed single-stream QPS from the same concurrency=1 warm loop, explicitly labelled as descriptive rather than server capacity;
- process-tree peak RSS, index bytes, model bytes, cache bytes, and GPU VRAM when applicable.

The reference run is offline, with concurrency, process count, ONNX/BLAS thread counts, provider,
CPU/GPU, OS, and power mode fixed and recorded. Repeat complete deterministic benchmark passes and report
paired per-query deltas with 95% confidence intervals. A candidate is promoted only when the
predeclared primary metric improves beyond uncertainty and all accuracy/latency/resource guardrails
remain satisfied. Latency budgets are calibrated from the first reference-machine runs rather than
invented before measurement.

The warm single-stream profile loops real queries for at least 60 seconds and at least 1,024
completed requests before reporting tail percentiles and descriptive single-stream QPS. A later server
profile may use declared arrival rates and report load-capacity QPS with p99; it is outside P0-P4. ANN
experiments publish quality-latency, quality-QPS, and quality-index-size Pareto curves rather than a
single tuned number.

Statistical comparisons use paired per-query scores, paired/bootstrap 95% confidence intervals,
and a paired randomization/permutation test. Multiple candidate comparisons apply Holm correction.
Correlated paraphrases and source clusters use cluster bootstrap. Reports include effect size and
taxonomy slices; a p-value alone never promotes a candidate.

## 12. Baseline experiment ladder

Only one principal variable changes per comparison:

```text
E0  BM25
E1  dense exact cosine
E2  BM25 + dense + RRF
E3  E2 + cross-encoder reranker
E4  metadata filter/boost
E5  query-type routing / query-dependent fusion
```

Every run stores:

- code commit and dirty-state flag;
- source, evidence, corpus-variant, benchmark, and config hashes;
- package lock and Python version;
- model identifiers, exact revisions, dimensions, licenses, and execution provider;
- hardware manifest and measurement protocol;
- per-query ranked results, per-stage timings, errors, and resource evidence;
- standard TREC run plus metrics report.

Later SPLADE, ColBERT, ANN, query expansion, learning-to-rank, caching, and fine-tuning enter only to
address measured failure classes and must beat the frozen E2/E3 controls.

## 13. Storage evolution and SQL access point

Portable immutable artifacts remain the source of truth for benchmark exchange and reproduction:

```text
JSONL corpus / queries / evidence units
TSV or TREC qrels
TREC run files
JSON manifests and metrics
NumPy vectors for the first exact baseline
```

SQL is an operational catalog and scalable retrieval backend, not a replacement for exportable
benchmark artifacts.

Planned logical tables:

```text
source_snapshots
documents
source_spans
evidence_units
corpus_variants
chunks
chunk_evidence
models
embeddings
benchmark_versions
queries
query_variants
qrels
evidence_rubrics
runs
run_results
stage_timings
resource_measurements
```

Storage stages:

1. file artifacts plus exact NumPy search establish the correctness oracle;
2. SQLite may provide a local SQL run/catalog view without introducing a service;
3. PostgreSQL + pgvector becomes the mature multi-user corpus/index backend when persistence,
   filtering, concurrent experiments, or corpus scale justify it;
4. pgvector exact search must first match the NumPy oracle; HNSW/IVFFlat are then separate ANN
   experiments with measured recall loss, build time, memory, and latency.

pgvector is appropriate because it keeps metadata and vectors in an ACID relational store and
supports exact search plus HNSW/IVFFlat. Its own documentation states that exact search has perfect
recall by default and ANN trades recall for speed; therefore ANN cannot silently replace the exact
baseline.

The stable boundary is records and IDs, not a premature class hierarchy. File and SQL adapters are
introduced only when both are real implementations and share parity tests.

## 14. Delivery gates

1. **Source gate** — pinned source, license, manifest, clean checkout, and independent location.
2. **Extraction gate** — deterministic Sphinx evidence units; explicit exclusion/error report;
   every unit is source-citable.
3. **Chunk gate** — all variants map to evidence units; structural integrity tests for tables,
   substitutions, includes, tabs, code, and admonitions.
4. **Benchmark gate** — versioned data card, pooled graded judgments, split/leakage audit, metric
   definitions, and TREC/BEIR compatibility. V0 additionally requires 160 double-annotated and
   adjudicated topics, the declared 96/64 dev/hidden split, agreement statistics, qrels pooling
   saturation evidence, hard-negative provenance, raw run receipts, and a statement that the pilot
   supports pipeline validation rather than a SOTA claim.
5. **Baseline gate** — E0–E3 reproduce from manifests and emit identical rankings within the
   declared numerical tolerance.
6. **Performance gate** — stage-level latency/resource report on the reference machine.
7. **Promotion gate** — frozen test comparison, uncertainty/effect report, guardrail checks, and an
   independent sub-agent delivery review.
8. **Integration gate** — only after the lab is independently accepted; no mainline integration is
   part of the current plan.

## 15. Remaining promotion work

P0-P4 provide an Agent-provisional experimental baseline. Depth-50 pooling, saturation,
hard-negative coverage, statistical diagnostics, no-answer diagnostics, and CLI gating have each
received fresh independent Agent approval. Promotion now requires:

1. have a human review every Agent disagreement, every no-answer candidate, and the declared
   stratified agreement sample, recording corrections in the versioned receipt;
2. freeze the corrected Seed50 gold, regenerate derived qrels for all four chunkers, and rerun the
   final E0-E3 matrix without using Seed as a SOTA claim;
3. only then design and collect the formal 160-topic V0 with two human annotators, adjudication, and
   the preregistered 96/64 dev/hidden split.

SQL/pgvector, ANN, generation, LLM judges, and main-Agent integration remain later experiments, not
implicit parts of this promotion step.

## References

- Odoo documentation source: https://github.com/odoo/documentation
- TREC qrels and relevance judgments: https://trec.nist.gov/data/qrels_eng/
- TREC RAG track: https://trec-rag.github.io/
- TREC 2025 RAG guidelines: https://trec-rag.github.io/annoucements/2025-track-guidelines/
- BEIR: https://github.com/beir-cellar/beir
- MTEB: https://github.com/embeddings-benchmark/mteb
- RAGBench / TRACe: https://arxiv.org/abs/2407.11005
- RAGAS metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- RAGFlow: https://github.com/infiniflow/ragflow
- Haystack DocumentSplitter: https://github.com/deepset-ai/haystack/blob/main/haystack/components/preprocessors/document_splitter.py
- Unstructured title chunking: https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/chunking/title.py
- pgvector: https://github.com/pgvector/pgvector
