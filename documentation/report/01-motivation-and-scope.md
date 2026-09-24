# 1. How far can efficient self-play go?

An AlphaZero-style chess player improves by searching its own games, learning from the resulting positions, and
repeating that cycle with a stronger network. Each step consumes compute. Search creates targets, training absorbs
them, and evaluation must distinguish real progress from noise. When all three share one eight-GPU node, a faster
model forward or a cheaper search matters only if it produces stronger play sooner.

This study asks how strong that loop can become under a very limited compute budget when the entire system is
engineered for efficiency. The selected model was trained from random initialization using self-play, without human
games or pretrained chess weights. It has 6.3 million parameters. Under the project's fixed-node Stockfish 13
calibration it measured **1,658 benchmark Elo without search** and **3,251 benchmark Elo at 100,000 searches per
move**. The previous four-day training baseline trails the final recipe by approximately **74 Elo** in an
estimator-matched 64-search plateau comparison. These numbers describe the stated match protocols; they are not
FIDE ratings or unrestricted-engine rankings. The full intervals and artifacts appear in
[the results chapter](07-final-run-results.md).

The most useful finding is the interaction among decisions. Search determines which targets are worth paying for.
Replay determines which of those targets the learner sees again. The network representation determines both what
can be learned and how quickly positions can be evaluated. Batching, native search, and TensorRT make enough games
possible for those choices to matter at all. A change that improved one local proxy sometimes worsened the complete
learning loop.

## Scope and contributions

Chess is the research subject. The same runtime supports Go on 7×7 and 9×9 boards, and a small-board baseline
[validated the shared platform](../benchmarks/go-7x7-training-baseline-2xrtx3060-20260810/README.md). Go also
supplied ideas, notably KataGo's fast and full searches. The project briefly considered small-board Go as a cheaper
place to tune parameters for chess. The basic loop worked, but its large first-player advantage, shorter games,
rapidly learned value target, and apparent need for different tuning made that transfer unattractive. This is the
project owner's qualitative rationale, not a controlled cross-game result. Go receives no separate strength claim
in this report.

The report contributes:

1. A complete, reproducible chess self-play run whose readable recipe starts at
   [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) and whose selected checkpoint,
   evaluation, and hashes are frozen in the [final result record](../results/final-chess-run.md).
2. Substantial investigations of search allocation, graph search, inference caching, policy representation,
   model sizing, replay, restart states, resignation, auxiliary targets, and quantized serving. Each conclusion is
   bounded by the workload and evidence actually measured.
3. An account of the throughput path that made the experiment feasible: native C++ game and tree ownership, batched
   GPU inference, TensorRT deployment, columnar replay, and persistent distributed training.
4. Three failures with transferable lessons about self-play targets, deployment correctness, and model promotion.

The final recipe combines many changes. Its strength establishes the outcome of the assembled system, not an
isolated Elo contribution for every retained component. The report uses paired games for playing-strength claims,
keeps throughput and target-fidelity measurements attached to their own protocols, and labels owner recollections
when original result artifacts are unavailable.

## Reading the report

[Chapter 2](02-methodology-and-evidence.md) gives the compact measurement rules. [Chapter 3](03-system-and-methods.md)
shows the learning loop. The investigation chapters then follow the three central choices: how to spend search,
what the network predicts, and which positions become training data. A short systems chapter explains how those
choices were made affordable. The three failure studies lead into the integrated recipe and the final measured
outcome. Detailed implementation history remains in the linked repository evidence rather than the paper's
narrative.
