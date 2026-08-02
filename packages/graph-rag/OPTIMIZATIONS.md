# GraphRAG Optimization Roadmap

This file tracks retrieval improvements worth evaluating after the initial implementation.

## Current priorities

- [x] Use the embedding model's tokenizer during document chunking.
- [x] Add exact-first, embedding-fallback entity linking.
- [x] Extract and link query relationships against graph edge labels.
- [ ] Weight Personalized PageRank seeds by entity-link similarity.
- [x] Fall back to vector retrieval when a query produces no graph seeds.

## Evaluation

- [ ] Create a representative question set with expected entities, documents, and chunks.
- [ ] Measure entity-link recall and chunk Recall@5 and Recall@10.
- [ ] Include both single-hop and multi-hop questions.

## Retrieval quality

- [ ] Add conservative synonym edges or entity canonicalization during indexing.
- [ ] Fuse graph-ranked and vector-ranked chunks using Reciprocal Rank Fusion.
- [ ] Rerank the fused candidates before answer generation.
- [ ] Compare sum, maximum, and length-normalized chunk score aggregation.

## Implementation notes

- Keep normalized exact entity matches as the highest-confidence path.
- Persist or cache graph-node embeddings instead of recomputing them per question.
- Calibrate entity similarity thresholds against the evaluation set.
