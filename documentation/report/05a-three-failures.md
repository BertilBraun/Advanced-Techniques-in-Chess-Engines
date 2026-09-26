# 6. Three failures that changed the method

Three shortcuts looked useful until their effects reached the games themselves. Skipping expensive endgames
withheld the examples needed to learn them. Reusing a compiled inference engine changed the model's predictions.
Promoting by training loss selected a player that fitted replay better but played worse. Each failure changed how
the system decides whether an apparent improvement is safe to use.

## Late-game target poisoning

The self-play design once reduced search near the end of long games. The intent was to keep noisy positions from
exceptionally long endgames from dominating replay. If those positions would not become training targets, spending
full-search compute on them seemed wasteful, so the tail used cheap searches instead. Only full-search positions
were admitted as primary training rows, however, so the cheap-search tail selectively removed the conversion phase
from the training set. The resulting policy was weakest exactly where it needed to turn an advantage into a terminal
result. Games wandered to the ply cap more often, and the cutoff then assigned one terminal value to every earlier
training row in the trajectory.

This created a closed feedback loop. Missing searched endgames produced weak late-game play; weak play produced
drawn-out, unconverted games; the cutoff supplied an unreliable target; and that target reinforced the same policy
and value errors. During the most affected early interval, roughly one ply in seven was structurally ineligible for
training, 27--32% of games reached the cap, and about 36--38% of admitted rows inherited the value assigned to a cut
game. The corruption was therefore not confined to the last position: the final target propagated back through the
entire recorded trajectory. By the time the active replay buffer looked healthy, the damaging rows had already been
evicted, while their effect on the weights and subsequent self-play distribution could remain.

The repair happened in two stages. First, the cutoff stopped relying on the material heuristic and obtained a value
from one full search at the final cut position. A controlled continuation study supported that choice: on 2,282
early cut positions, the searched root value reduced Brier error from 0.491 to 0.444 and cross-entropy from 0.851 to
0.756 relative to the material target. The same ordering held on a later set of 1,144 positions. Later, the forced
cheap-search tail was removed altogether, returning properly searched endgame positions to replay. The second change
closed the data hole that a better cutoff value alone could not repair. Together, the changes restored both useful
endgame examples and a better estimate for games that still had to be stopped.

![Late-game target poisoning feedback loop and its two-stage repair](figures/late-game-poisoning-feedback-loop.svg)

Figure 7: A searched cutoff value improves labels for unfinished games; restoring fully searched endgames also
returns the examples needed to learn conversion.

Deciding which moves deserve search also decides what the network gets to learn. Discarding one part of the game
can make the player weakest there, causing later self-play to repeat the same blind spot.

## When a successful TensorRT refit changed the model

Rebuilding a TensorRT engine after every training block is expensive, so the deployment pipeline builds a template
and replaces its weights through refitting. Quantized computation also uses scale factors to map integer values
to their numerical ranges. In one template, many of those scales happened to be equal, and TensorRT's higher
optimization levels compiled around that equality. Later training made the scales different. Refitting still
accepted every replacement and reported success, but the engine no longer reproduced the network's predictions.

Comparing predictions on 516 real positions exposed the defect. KL divergence measures how much two probability
distributions disagree, with zero meaning agreement. The faulty engine's move probabilities differed from the
source network by a KL divergence near 1.05, and repeated refits
of the same inputs changed individual logits by 10--15. Building at a lower optimization level, or first making the
template's scale constants distinct, reduced legal-policy KL to about 0.0012 and made repeated refits deterministic.
The failure required three conditions: equal scales in the build source, optimization level four or higher, and a
later refit that made those scales unequal.

Template construction now separates equal quantization scales before optimization and uses a safer default
optimization level. Each exported engine is then compared with its source network on real positions: does it prefer
the same legal move, assign similar probabilities to alternatives, and preserve the win/draw/loss prediction?
Top-one agreement, policy KL, and WDL error measure these three aspects of the behaviour that search consumes.

The practical lesson is to test the exported player, not just the export operation. A successful engine build or
refit cannot substitute for checking that it still predicts the intended moves and values.

## Promotion from incomparable training losses

The progressive controller originally promoted a larger candidate when its smoothed training loss caught up with
the active model. Loss looked like a cheap proxy for readiness: it was already measured during training, whereas a
full-search candidate match consumed additional evaluation compute and time. That comparison ceased to be meaningful
once the candidate received extra optimizer quanta over the same replay distribution. More presentations gave it a
systematic advantage on replay loss without proving equal playing strength. The loss gate promoted a candidate whose
deployed artifact passed inference-fidelity checks but was still about 270 Elo weaker in play.

Promotion now tests playing strength directly. The candidate plays paired head-to-head matches
against the active deployment artifact and must score at least 0.48 in two consecutive completed evaluations. A
failing score resets the sequence; a failed or cancelled match contributes no evidence. Candidate-start timing,
extra catch-up training, and promotion are separate controls. Function-preserving growth is likewise a separate
attempt to remove the larger model's initial relearning deficit, not a substitute for the playing-strength gate.
Chapter 7 states the retained candidate-start and promotion procedure.

Training loss remains valuable for optimization diagnostics. It is not a promotion criterion when the compared
models have seen different numbers of presentations or when the artifact intended for deployment can be tested
directly.
