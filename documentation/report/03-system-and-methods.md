# 3. Background and system design

## Learning through search

AlphaZero learns a chess player without examples of human play [1]. Its network has two jobs: the *policy* assigns
probabilities to moves, while the *value* estimates the outcome of a position. Monte Carlo tree search (MCTS) brings
these predictions together. It explores moves using both their policy probabilities and the results found so far,
balancing promising continuations against less-explored alternatives. At a new leaf, the network evaluates the
position; that value is backed up along the path to inform subsequent exploration.

Search can therefore challenge the network's first impression. A promising move may reveal a strong reply for the
opponent, while a less obvious move may lead to better positions. The root visit distribution becomes a policy
target, and the completed game's outcome teaches the value prediction. Training folds this experience back into
the network. Better predictions then guide later searches towards more useful continuations, sustaining a cycle
of search, self-play, and learning.

This is the central opportunity under limited compute: spend search to discover improvements, then learn enough
from those improvements that the next search starts from a stronger player. It also explains why speed alone is
insufficient. Cheap searches that mostly repeat the network's initial preference may produce many positions but
little new policy information. Conversely, very deep searches can make good targets too expensive to supply in
sufficient quantity. The experiments in Chapter 4 explore this balance.

## Related work

AlphaZero establishes the self-play learning framework [1]; KataGo shows how substantially its compute requirements
can be reduced through changes to search, training, and network architecture [2]. KataGo is the closest practical
precedent for this study's efficiency focus. Its fast/full search schedule, auxiliary objectives, and self-play
methods [7] motivated several investigations here. Their usefulness still depends on the game: completing more
long Go games and supplying more searched chess positions need not favour the same allocation of compute.

Several narrower lines of work address where that compute should go. Dynamic simulation MCTS studies when to stop
search [3], and targeted search control starts self-play from archived states to explore beyond ordinary opening
trajectories [5]. Prioritized
experience replay changes which stored examples are learned from again [4], while Monte Carlo graph search shares
work across paths reaching the same state [6]. These ideas motivate the allocation, replay, restart, and reuse
experiments below. Our contribution is an integrated, limited-compute chess study of those choices: the design that
worked together, the measurements behind it, and the alternatives that did not repay their cost.

## One learning cycle

Figure 1 follows a model through the implemented learning cycle. Python orchestration publishes its weights to the
self-play actors and schedules training and evaluation. Each actor advances many chess games in native C++.
Whenever search reaches positions needing evaluation, the actor groups them into batches for TensorRT, the
optimized GPU inference engine.
Returned policies and values allow the waiting searches to continue, update their visit statistics, and eventually
choose the next moves. The expensive interaction is therefore between native search and batched inference, not
between Python and each individual tree traversal.

![Python coordination, native self-play, batched TensorRT inference, replay, training, and evaluation feedback](figures/learning-loop.svg)

Figure 1: A published model guides batched native self-play; searched positions become replay targets for the
trainer. Paired evaluation measures the published checkpoint and informs model-promotion decisions.

When a game ends, its outcome completes the targets for its recorded positions. The replay pipeline converts these
trajectories into stored examples, and the trainer draws batches from that accumulated experience. After a block of
optimizer steps, the updated model is published back to the actors. Some self-play continues during training, so
data production and learning overlap rather than alternating between an idle trainer and idle actors.

Evaluation runs alongside this loop using the native chess engine to play paired matches. It measures progress
without adding those matches to self-play training. Its feedback also determines when a larger candidate is ready
to replace the active model. In Figure 1, the solid arrows carry positions, predictions, training data, or models;
the dashed return path carries this evaluation feedback to Python orchestration.

## Why the search loop stays in C++

The language boundary follows the frequency of the work. A single played move needs hundreds or thousands of tree
traversals, each involving board updates, move generation, selection, and backup. Running this inner loop through
Python makes interpreter and boundary-crossing overhead recur at every search step. Keeping the complete loop in
C++ lets actors retain trees and buffers, advance independent games while evaluations are pending, and feed the
GPU without a Python callback for each leaf.

Python remains useful at the coarser scale. It coordinates workers and model publication, manages replay, and runs
training through PyTorch and its distributed libraries. Those tasks benefit from the existing ecosystem and ease
of experimentation without putting Python on the path of every simulation. The split is an efficiency choice:
native code supplies enough searched games for learning, while Python keeps the surrounding training system
manageable. Chapter 5 measures the resulting throughput and the bottlenecks that remain.

## Chess representation and outputs

The network sees the board, recent history, and rule-relevant state. It predicts a move policy and the probabilities
of winning, drawing, or losing (WDL). Search uses those two outputs to choose moves and to create training targets. The
retained model shares a convolutional representation between its outputs and scores moves by origin and destination;
training-only auxiliary heads are removed from the serving artifact. Section 4.3 explains the representation
choices, including the 1,880-action interface; Chapter 7 gives the selected model shape.

## Search and self-play

Self-play searches each played position with neural policy priors and value estimates. The visit budget grows in
stages as training progresses, but is fixed across positions within a stage. Each search produces both a move and a
policy target for learning. Section 4.1 explains the search rules and why the tested ways of varying the budget
within a stage were not retained.

Many games run concurrently so their neural evaluations can share GPU batches. Native workers retain search trees
between moves, and TensorRT serves the batches. Searching several leaves of one tree at once can fill a batch when
fewer independent games are ready, but the leaves then see partly stale search information. Sections 4.1 and 5
measure that quality–throughput tradeoff and the worker topology.

Games start either from shallow random openings or from archived self-play positions worth revisiting. This keeps
ordinary opening-to-endgame games in the stream while spending some games on consequential branches that search
found difficult. Section 4.2 explains how restart states are selected and how resignation is checked against games
allowed to continue to their natural result.

## Replay and materialization

After a game finishes, its searched positions become replay rows containing the position, legal actions, search
policy, and outcome target. The workers publish complete trajectories; materializers convert them to a circular
memory-mapped store. Policies stay sparse until a training batch is assembled. Admission checks prevent incomplete
or rejected games from being counted as usable data.

The trainer sees a mix of broadly sampled positions and positions where search substantially changed the network's
move preference. Replay grows as more games arrive, while the amount of optimizer work is tied to newly admitted
positions. Section 4.2 follows these distinct choices—where games begin, what becomes a row, and which rows recur
in training—and gives the retained settings.

## Training

Training minimizes errors in the search policy and the game outcome. Two auxiliary tasks use information from a
completed game: predicting the next move's searched policy and the remaining game length. They help train
the shared network but do not run during self-play. Section 4.2 explains when those future-dependent targets exist.

Eight persistent trainer processes use distributed data parallelism (DDP): each GPU trains on part of the batch,
and their gradients are combined for the model update. They train in 500-step blocks with a global batch of 2,048 while some
self-play workers continue producing games. A block followed by publication is a *generation*; its saved model is a
*checkpoint*. Overlap matters: saving search work does not necessarily shorten a generation if training was already
the limiting step. The adaptive-stopping test in Section 4.1 measures that distinction.

## Progressive models and publication

Training starts with a small network: its speed produces more searched games while the learner has little use for
extra capacity. A larger candidate trains on the same replay alongside the active model and takes over only after
paired matches show it has caught up. Growing a trained model into a larger one was also tested, but is not the
ordinary promotion path. Section 4.3 examines this choice; Chapter 7 specifies the plateau and match gates.

Publication saves a recoverable training checkpoint and a smaller inference artifact. The deployed path exports
quantization-aware-trained weights to ONNX and refits a TensorRT engine for self-play and evaluation. The engine's
outputs are checked against the source model: a successful build alone does not prove that it plays the same moves.
Chapter 6 explains the failure that prompted this check.

## Evaluation

During training, fixed-position metrics and short paired-opening matches show whether learning is progressing.
Larger matches at several search budgets test the final model; Table 1 in Chapter 2 reports their results. Each
opening is played with both colors against a fixed Stockfish node limit. Appendix B explains the rating calculation
and gives supporting evaluation details.

The run preserves the configurations, engine and data identities, checkpoints, and evaluation records needed to
interpret and reproduce those measurements. Appendix D lists the public and archived artifacts.
