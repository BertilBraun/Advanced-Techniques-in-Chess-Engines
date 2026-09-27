# 6. Three failures that changed the method

Three failures exposed dependencies that local optimization metrics did not capture: endgame sampling altered
the reliability of value targets, TensorRT refitting changed the deployed policy, and loss-based promotion confused
replay fit with playing strength. Their causes and corrections illustrate why data generation, deployment, and
model selection must be evaluated as parts of the learning process.

## Late-game target poisoning

To prevent noisy positions from exceptionally long endings from dominating replay, an earlier self-play design
excluded the late-game tail from primary training targets and reduced its search budget. This saved compute on
positions that would otherwise be discarded, but systematically removed examples of endgame conversion. With
little training on those positions and only shallow search during play, the model frequently failed to convert
advantages before the ply cap. In the affected run, the preceding cheap search's root value supplied the outcome
target for the recorded trajectory, after reversing the side-to-move perspective.

![Late-game target poisoning feedback loop and two corrective interventions](figures/late-game-poisoning-feedback-loop.svg)

Figure: A searched cutoff value improves labels for unfinished games; restoring fully searched endgames also
returns the examples needed to learn conversion.

The interaction was self-reinforcing: inadequate conversion produced more capped games, whose unreliable targets
further degraded the policy and value estimates needed to finish them. During the most affected interval,
approximately one ply in seven was ineligible for training, 27--32% of games reached the cap, and 36--38% of admitted
rows inherited a cutoff target. Consequently, an error introduced at the end of a game affected training throughout
its recorded trajectory. Replay turnover removed the original rows but did not necessarily remove their influence
on the model or the self-play distribution.

An earlier target-comparison study had already favoured searched values over the material baseline: on 2,282
early-cut positions, the cut-position search achieved Brier error 0.444 and cross-entropy 0.756, versus 0.491 and
0.851 for material. A later set of 1,144 positions showed the same ordering. These measurements compare target
estimators, not the complete conversion repair. The initial implementation reused the preceding move's search;
under forced late-game cheap search, that bootstrap still came from a shallow search at the adjacent position.

The subsequent correction performed a dedicated full search at the actual cutoff, supplying both its root value
and a searched policy target for that position. Restoring searched endgame positions removed a known gap in
training coverage (Figure \ref{fig:late-game-poisoning-feedback-loop}). Conversion recovered after several concurrent
changes, including discounting, ply-cap scheduling, and resignation settings, so their individual contributions
were not isolated.

## Prediction drift under TensorRT refitting

The deployment pipeline refits a compiled TensorRT template after each training block to avoid rebuilding the
engine. In TensorRT 10.14.1.48, a failing template's equal quantization-scale constants allowed optimizations that became invalid when
subsequent training produced unequal scales. TensorRT accepted all replacement weights and reported a successful
refit, yet the resulting engine no longer reproduced the source network's predictions.

On 516 real positions, the faulty engine's legal-policy KL divergence from the source network was approximately
1.05, and repeated refits with identical inputs changed individual logits by 10--15. Lowering the optimization
level or making the template's scale constants distinct before compilation reduced KL divergence to approximately
0.0012 and restored deterministic refitting. The defect required the combination of equal source scales,
optimization level four or higher, and a subsequent refit with unequal scales.

Template construction now separates equal quantization scales before optimization and uses a lower default
optimization level. Deployment validation compares top-one legal-move agreement, policy KL, and WDL error against
the source network. These checks test the predictions consumed by search, independently of whether the compiler
reports a successful build or refit.

## Promotion from incomparable training losses

The progressive controller initially used smoothed training loss as a low-cost proxy for promotion readiness,
avoiding additional searched matches. Extra catch-up training, however, gave the larger candidate more optimizer
quanta on the same replay distribution. Its lower loss therefore reflected unequal training exposure as well as
model quality. The gate promoted a candidate approximately 270 Elo weaker than the active player, despite its
deployment artifact passing inference-fidelity checks.

Promotion now requires a score of at least 0.48 in two consecutive paired matches against the active deployment
artifact. A failing score resets the sequence; failed or cancelled matches do not count. This separates the
optimization objective from the replacement decision: training loss guides candidate fitting, while direct play
determines whether it can replace the active model. Appendix \ref{app:D} specifies candidate timing and catch-up training.
