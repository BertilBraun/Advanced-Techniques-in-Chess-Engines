# 1. How far can efficient self-play go?

An AlphaZero-style chess player [1] improves by searching its own games, learning from the resulting positions, and
repeating that cycle with a stronger network. Search creates targets; replay decides which targets persist; training
absorbs them; and evaluation must distinguish progress from noise. Under a limited budget, improving any one step
matters only if the complete loop produces stronger play within the available time.

This study asks how strong that loop can become on a single eight-GPU node when the system is engineered for
efficiency. Over **2.5 days** of training from random weights on searched self-play games, the run produced a
6.32-million-parameter model. It reached **1,658 benchmark Elo without search**
and **3,251 benchmark Elo at 100,000 searches per move**—estimated at under five seconds of thinking time—against the project's fixed-node Stockfish 13 ladder.
Table \ref{tab:02-methodology-and-evidence-1} gives the match results; Chapter \ref{sec:07-final-run-results} follows the model's improvement across search budgets.

The result depends on the whole learning loop. The network's representation affects both what it can learn and how
quickly it can supply search. Native search, batched inference, and TensorRT make enough searched games available
for training; replay and training must then turn those games into stronger play. Several plausible optimizations
improved a local metric without improving that complete loop.

## Scope and contributions

Chess is the subject of the study. The runtime also supports Go, and KataGo's fast and full searches inspired one
of the approaches tested here [7]. We briefly considered 7×7 and 9×9 Go as cheaper settings for tuning chess
hyperparameters. In those exploratory games, we observed a strong first-player advantage, short trajectories, and
a value target that learned quickly. Useful Go tuning appeared unlikely to transfer directly to chess, so we kept
chess as the focus.

We examine the choices that shaped the final recipe: search allocation, graph search and caching, policy
representation, model sizing, replay and restart states, resignation, auxiliary targets, and quantized inference.
The report also traces the throughput needed to supply searched games and studies three failures with transferable
lessons about self-play targets, deployment fidelity, and model promotion.

A live chess demonstration is also available [12].

## Roadmap

Chapter \ref{sec:02-methodology-and-evidence} presents the chess evaluation.
Chapter \ref{sec:03-system-and-methods} explains the AlphaZero learning principle and the system that implements it.
Chapter \ref{sec:04-research-investigations} examines search, replay, and network design, followed by throughput engineering in
Chapter \ref{sec:05-systems-optimization}. The three failure studies in Chapter \ref{sec:05a-three-failures}
lead into training progress in Chapter \ref{sec:06-final-chess-recipe} and playing strength in
Chapter \ref{sec:07-final-run-results}. Appendix \ref{app:D} specifies the final recipe.
