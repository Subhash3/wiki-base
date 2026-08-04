# GraphRAG Optimization Roadmap

This file tracks retrieval improvements worth evaluating after the initial implementation.

## Current priorities

- [x] Use the embedding model's tokenizer during document chunking.
- [x] Add exact-first, embedding-fallback entity linking.
- [x] Extract and link query relationships against graph edge labels.
- [ ] Weight Personalized PageRank seeds by entity-link similarity.
- [x] Fall back to vector retrieval when a query produces no graph seeds.

## Evaluation

- [x] Create a development question set with expected chunks and answer facts.
- [x] Measure chunk hit rate, Recall@K, MRR, fallback use, and request time.
- [ ] Include both single-hop and multi-hop questions.

## Retrieval quality

- [x] Add conservative pgvector synonym edges during indexing.
- [ ] Fuse graph-ranked and vector-ranked chunks using Reciprocal Rank Fusion.
- [ ] Rerank the fused candidates before answer generation.
- [ ] Compare sum, maximum, and length-normalized chunk score aggregation.

## Implementation notes

- Keep normalized exact entity matches as the highest-confidence path.
- [x] Persist graph entity and relationship embeddings for pgvector concept search.
- [x] Prefer relationship matches connected to already-linked query entities.
- [x] Reject degenerate OpenIE self-facts before graph construction.
- Calibrate entity similarity thresholds against the evaluation set.
