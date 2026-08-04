# GraphRAG retrieval benchmark

`graphrag.json` is the development retrieval benchmark used to compare Lite and Pro.
Each case identifies the wiki base, question, expected evidence, and optional answer facts.

A relevant chunk can be identified by UUID, document name, a stable content fragment, or a
combination of them. Content fragments make labels resilient when documents are ingested
again and receive new chunk UUIDs.

Run both retrieval modes against a running Wiki Base API:

```bash
uv run wiki-base-benchmark benchmarks/graphrag.json \
  --run-name before-entity-guided-openie
```

The default report is written to `benchmarks/graphrag.results.json`. Use `--output` to keep
named baseline reports:

```bash
uv run wiki-base-benchmark benchmarks/graphrag.json \
  --run-name before-entity-guided-openie \
  --output benchmarks/baselines/before-entity-guided-openie.json
```

Use `--mode lite` or `--mode pro` to run one mode. By default, the runner executes both
sequentially and records:

- ranked chunk IDs and previews;
- the actual retrieval strategy, including vector fallback;
- hit rate and recall at the requested limit;
- mean reciprocal rank;
- request duration and API errors.

Update relevance labels when the development documents or wiki-base IDs change. A case may
contain several `relevant_chunks`; all of them contribute to recall.

The valid pre-improvement report is stored in
`baselines/before-entity-guided-openie.json`. Lite achieved `0.875` mean recall and `0.750`
MRR; Pro achieved `0.750` mean recall and `0.458` MRR.

The first entity-guided OpenIE report is stored in
`baselines/after-entity-guided-openie.json`. Pro recall remained `0.750`, while MRR fell to
`0.396`; that regression is the baseline for the relationship-context and graph-cleanup
tuning pass.
