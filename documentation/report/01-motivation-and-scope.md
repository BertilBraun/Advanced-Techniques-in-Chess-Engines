# 1. How far can efficient self-play go?

An AlphaZero-style chess player [1] improves by searching its own games, learning from the resulting positions, and
repeating that cycle with a stronger network. Search creates targets; replay decides which targets persist; training
absorbs them; and evaluation must distinguish progress from noise. Under a limited budget, improving any one step
matters only if the complete loop produces stronger play within the available time.

This study asks how strong that loop can become on a single eight-GPU node when the system is engineered for
efficiency. Over **2.5 days** of training from random weights, using searched self-play rather than human-game
training targets, the run produced a 6.32-million-parameter model. It reached **1,658 benchmark Elo without search**
and **3,251 benchmark Elo at 100,000 searches per move** against the project's fixed-node Stockfish 13 ladder. A
matched-estimator comparison places it about **74 Elo** above the previous training baseline at a 64-search
plateau. These are results under specific match protocols, not FIDE ratings or unrestricted-engine rankings. The
terminal games and intervals appear in Chapter 8 and Appendix B.

The central lesson is that the choices interact. Search determines which targets are worth paying for; replay
determines which targets the learner sees again; and representation affects both learning and inference cost. Native
search, batched inference, and TensorRT provide the throughput that makes those choices consequential. Several
plausible optimizations improved a local proxy without improving the complete learning loop.

## Scope and contributions

Chess is the subject of the study. The runtime also supports 7×7 and 9×9 Go, which helped validate the shared
platform and supplied ideas such as KataGo's fast and full searches [7]. Small-board Go was briefly considered as a
cheaper setting for tuning chess hyperparameters. Its first-player advantage, shorter games, rapidly learned value
target, and apparent need for different tuning made that transfer unattractive. This is the project owner's
qualitative rationale, not a controlled cross-game finding; the report makes no Go strength claim.

The report presents the training recipe and terminal evaluation, then examines the choices that shaped them:
search allocation, graph search and caching, policy representation, model sizing, replay and restart states,
resignation, auxiliary targets, and quantized inference. It also explains the throughput path needed to supply
searched games and studies three failures with transferable lessons about self-play targets, deployment fidelity,
and model promotion. The readable, living recipe begins at `chess-final-config.yaml` [10]; the reported checkpoint
is identified separately from that configuration.

The final recipe combines many changes. Its strength establishes the outcome of the assembled system, not an
isolated Elo contribution for each component. We use paired games for playing-strength claims, keep throughput and
target-fidelity measurements tied to their own protocols, and distinguish preserved evidence from owner
recollection when historical result artifacts are missing.

A live chess demonstration is available [12], but its games are not part of the evaluation protocol.

## Reading the report

Chapter 2 defines how evidence is interpreted; Chapter 3 presents the learning loop. The investigation chapters
then ask how to spend search, what the network predicts, and which positions become training data. A short systems
chapter explains how enough games were produced. Three failure studies lead into the integrated recipe and final
evaluation. Implementation records are available in the project repository [10], but the paper states the methods
and results needed for its own conclusions.
