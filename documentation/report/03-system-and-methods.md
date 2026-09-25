# 3. System and training method

## One learning cycle

The system improves by repeatedly turning played games into a stronger player. A published network guides search
in many self-play games; the resulting positions and search policies enter replay; the trainer learns from that
replay and publishes the next network. Evaluation checks whether the new player is actually stronger. Figure 1
shows this cycle and the components that carry it. Chapter 5 asks which parts of the cycle limit how much learning
can happen in a fixed time.

![Python coordination, native self-play, batched TensorRT inference, replay, training, and evaluation feedback](figures/learning-loop.svg)

Figure 1: A published model guides batched native self-play; searched positions become replay targets for the
trainer. Paired evaluation measures the published checkpoint and informs model-promotion decisions.

Native C++ plays the games and runs search so Python does not handle every position or tree operation. Python
coordinates workers, replay, training, publication, and evaluation. Both sides must interpret chess positions and
network outputs identically; tests check their action mapping, features, symmetries, tensor shapes, and output order.

## Chess representation and outputs

The network sees the board, recent history, and rule-relevant state. It predicts a move policy and the probabilities
of winning, drawing, or losing. Search uses those two outputs to choose moves and to create training targets. The
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

Eight persistent trainer processes, one per GPU, train in 500-step blocks with a global batch of 2,048 while some
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
