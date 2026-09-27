# Appendix A. Training diagnostics

The diagnostic traces cover the training sequence that produced checkpoint 1026: 817 quanta of 500 optimizer
steps at global batch 2,048, totaling 408,500 steps and 836,608,000 presentations. Checkpoint identifiers include
additional experimental branches, whereas these plots exclude the reverted INT8 conversion, invalid capacity
promotion, and subsequent larger-model continuation. Checkpoint 1026 was selected near the strongest 64-search
ladder region and was the last checkpoint in that region with the complete model, optimizer, QAT, ONNX, and
TensorRT artifacts retained.

![Training losses and learning rate through the selected checkpoint](figures/final-training-loss-and-rate-paper.svg)

Figure: Training losses and learning rate. Raw losses are overlaid with 11-quantum moving averages; the
dashed line marks the small-to-medium transition at 240,000 optimizer steps. Total loss reaches approximately
2.95 while WDL loss remains near 0.80. The learning rate declines
from approximately 0.10 to 0.0266. Labels identify the final values.

The exact counters underlying Chapter \ref{sec:06-final-chess-recipe} are 3,249,647 completed games and approximately 209.15 million net
materialized positions, with 16 million live rows at selection. Median trainer throughput was 16,977 samples/s
over 480 small-model quanta and 11,194 over 337 medium-model quanta. Figure \ref{fig:final-training-volume-and-throughput-paper} in Chapter \ref{sec:06-final-chess-recipe} shows these trajectories.
Node rental was $0.72/hour, totaling $43.20 for the 2.5-day run.

![Next-policy and remaining-game-length losses on separate scales](figures/appendix-auxiliary-losses.svg)

Figure: Next-policy and remaining-game-length losses on separate scales. Faint traces show recorded values;
solid curves show 11-quantum moving averages. The dashed line marks the small-to-medium transition, as in the
remaining optimizer-step plots.

![Mean pre-clipping gradient norm with the configured clipping threshold](figures/appendix-gradient-norm.svg)

Figure: Mean pre-clipping gradient norm in each 500-step training block. The dotted line is the configured
norm limit of 1.0. The mean rises above that limit during medium-model training; the optimizer clips individual
steps before applying them.

![Policy-only and 64-search training ladders over the final 2.5-day window](figures/appendix-policy-search-progress.svg)

Figure: Policy-only and 64-search training-ladder ratings over the final 2.5-day window. Faint traces show
individual evaluations and solid curves average eleven evaluations. Both improve over training; Table \ref{tab:02-methodology-and-evidence-1} reports
the separate final matches.

Replay occupancy, sampled-position age, and resignation calibration are shown with their corresponding methods
in Section \ref{sec:04b-data-and-replay} (Figure \ref{fig:appendix-replay-age} and Figure \ref{fig:appendix-resignation}).
