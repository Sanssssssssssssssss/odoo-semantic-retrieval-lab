# Seed50 provisional annotation guide

Seed50 is an engineering pilot, not a formal human gold set. Two agents annotate the same frozen topic briefs independently; a third agent adjudicates every difference. A later human review must cover every disagreement, every no-answer topic, and a stratified sample before the status can change from `provisional`.

## Evidence boundary

- Gold evidence must be an approved `SourceSpan` belonging to the pinned Odoo 19 `applications` corpus.
- `developer` and `administration` evidence is forbidden.
- A nugget is one atomic fact needed to answer the query. Mark it `required=true` if an answer is incomplete without it.
- Do not bind gold to a chunk ID. Chunk qrels are generated later from canonical spans.
- Keep distinct spans when a question needs multiple facts. Do not select a broad nearby section merely because it contains the right keywords.

## Four-level judgment

- `0`: irrelevant or contradicts the requested module/version/condition.
- `1`: topically related but does not supply answer evidence.
- `2`: useful answer evidence that is incomplete or must be combined with other evidence.
- `3`: direct, complete, and self-contained evidence for the associated nugget(s).

Unjudged candidates are not grade 0. Seeded grade 0/1 examples are useful but heterogeneous pooling will add the required hard negatives later.

## Answerability

An answerable topic needs at least one required nugget and at least one grade 2 or 3 judgment. A no-answer topic has no positive nugget and no grade 2/3 judgment. Its reason is exactly one of:

- `absent_from_corpus`
- `requires_live_instance`
- `wrong_version`
- `enterprise_or_excluded`
- `ambiguous_requires_clarification`
- `out_of_scope`

Do not convert ambiguity into an invented assumption. No-answer labels evaluate abstention separately and never enter ordinary ranking aggregates.

## Evidence topology

- `single_span`: one canonical source span is sufficient.
- `same_page_multi_span`: multiple non-equivalent spans from one source document are required.
- `cross_page`: required spans come from at least two source documents.

## Blindness and provenance

Annotators may read the topic briefs and the frozen corpus only. They must not read the other annotator's output. Each submission records its own annotator ID. The adjudicator reads both submissions, rechecks every cited source, and records a fresh decision; it must not resolve differences by blindly preferring one annotator.
