# Engineering Efficient Self-Play Chess

## Abstract

How strong can an AlphaZero-style chess system become under limited training compute when its entire learning loop
is engineered for efficiency? We train from random initialization through searched self-play on a single eight-GPU
node. The reported training path runs for 60 hours at $43.20 in node rental, excluding discarded experiments and
evaluation. We examine search allocation, replay and restart-state selection, policy representation, progressive
model sizing, quantized inference, and throughput engineering, including approaches that did not justify their cost.
Over those 2.5 days, the run ingests 3.25 million completed games and makes 836.6 million training presentations.
The resulting 6.32-million-parameter model reaches 3,251 benchmark Elo [3,206, 3,297] at 100,000 searches per
move on a fixed-node Stockfish 13 ladder. That strength belongs to the integrated system rather than to any
isolated component improvement.
