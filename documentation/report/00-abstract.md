# Engineering Efficient Self-Play Chess

## Abstract

How strong can an AlphaZero-style chess system become under limited training compute when its entire learning loop
is engineered for efficiency? We train from random initialization through searched self-play on a single eight-GPU
node for 2.5 days. The final 6.32-million-parameter model reaches 3,251 benchmark Elo [3,206, 3,297] at 100,000
searches per move on a fixed-node Stockfish 13 ladder. Training ingests 3.25 million completed games, involves
roughly 200 billion search-time neural-network evaluations, and makes 836.6 million training presentations. We examine
search allocation, replay and restart-state selection, policy representation, progressive model sizing, quantized
inference, and throughput engineering, including approaches that did not justify their cost. That strength belongs to
the integrated system rather than to any isolated component improvement.
