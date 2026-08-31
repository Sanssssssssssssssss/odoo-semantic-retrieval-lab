# Seed50 provisional annotation guide

Seed50 is an engineering pilot, not a formal human gold set. Two agents annotate the same frozen topic briefs independently; a third agent adjudicates every difference. A later human review must cover every disagreement, every no-answer topic, and a stratified sample before the status can change from `provisional`.

## Evidence boundary

- Gold evidence must be an approved `SourceSpan` belonging to the pinned Odoo 19 `applications` corpus.
- `developer` and `administration` evidence is forbidden.
- A nugget is one atomic fact needed to answer the query. Mark it `required=true` if an answer is incomplete without it.
- Do not bind gold to a chunk ID. Chunk qrels are generated later from canonical spans.
- Keep distinct spans when a question needs multiple facts. Do not select a broad nearby section merely because it contains the right keywords.

## Mechanical four-level judgment

Annotators record `topic_relevance` and the IDs of required atomic nuggets actually supported by the visible evidence. `grade` is not subjective; tooling derives it:

- `0`: not topically relevant; no nugget hit and no selected span.
- `1`: topically relevant but hits no required nugget; at least one topical span is selected.
- `2`: hits at least one, but not all, required nuggets.
- `3`: hits every required nugget.

For no-answer topics, required nugget hits are always empty, so visible topical material can be grade 1 but never a positive qrel. A stored grade that differs from the derived grade is invalid.

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

Annotators may read the topic briefs, the frozen atomic-nugget rubric, and the frozen corpus only. The rubric exposes nugget text but never canonical gold spans or system labels. They must not read the other annotator's output. Each submission records its own annotator ID. The adjudicator reads both submissions, rechecks every cited source, and records a fresh decision; it must not resolve differences by blindly preferring one annotator.
