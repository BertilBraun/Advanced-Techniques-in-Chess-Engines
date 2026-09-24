# Engineering Efficient Self-Play Chess

## Abstract

How strong can an AlphaZero-style chess system become under limited training compute when its entire learning loop
is engineered for efficiency? We study a model trained from random initialization through searched self-play on one
eight-GPU node. Its accepted training lineage spans 60 effective hours and ingests 3.25 million completed games,
about 209 million net replay positions, and 836.6 million training presentations. The selected 6.3-million-parameter
model reaches 3,114 benchmark Elo [3,065, 3,163] at 10,000 searches and 3,251 [3,206, 3,297] at 100,000 searches
on a fixed-node Stockfish 13 ladder. An estimator-matched 64-search comparison places its late training plateau
approximately 74 Elo above the previous baseline. We examine search allocation, replay and restart-state selection,
policy representation, progressive sizing, quantized inference, and throughput engineering, including approaches
that failed or did not justify their cost. The resulting strength belongs to the integrated system; component
proxies are not treated as isolated causal Elo gains. All ratings are protocol-specific benchmark estimates, not
FIDE or unrestricted-engine ratings.
