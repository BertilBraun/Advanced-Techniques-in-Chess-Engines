# Engineering Efficient Self-Play Chess

## Abstract

How strong can an AlphaZero-style chess system become under limited training compute when its entire learning loop
is engineered for efficiency? We train from random initialization through searched self-play on a single eight-GPU
node for 2.5 days. The resulting 6.32-million-parameter model reaches 3,251 benchmark Elo [3,206, 3,297] at
100,000 searches per move against a fixed-node Stockfish 13 ladder. The run ingests 3.25
million completed games, involves an estimated 195 billion search-time neural-network evaluations, and makes 836.6
million training presentations. We investigate search allocation, replay and restart-state selection, policy
representation, progressive model sizing, quantized inference, and throughput engineering. Alongside the retained
design, we document plausible alternatives that failed to improve the complete learning loop or did not justify their
cost. The reported strength is a result of the integrated system, not an isolated Elo gain attributable to any single
choice.
