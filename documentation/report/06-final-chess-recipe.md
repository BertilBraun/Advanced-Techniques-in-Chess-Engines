# 7. Training progress under limited compute

The final training run produced a 6.32-million-parameter chess model in 2.5 days on one node with eight
RTX 4070 SUPER GPUs and 80 logical CPUs. This chapter examines the progression in strength and the training volume
made possible by the integrated system. The complete recipe is specified in Appendix \ref{app:D};
`chess-final-config.yaml` [10] remains the entry point for reproduction.

## Progress across training campaigns

Figure \ref{fig:chess-ladder-progress-paper} compares the 64-search training ladders across five chess campaigns. The final recipe reached higher
playing strength within the reported training window.

![64-search ladder Elo across five chess training campaigns](figures/chess-ladder-progress-paper.svg)

Figure: Smoothed 64-search training ladders across five chess campaigns. The final curve ends at 2.5 days and
the preceding baseline at 3 days.

![Ingested games, replay positions, and trainer throughput](figures/final-training-volume-and-throughput-paper.svg)

Figure: Training volume and throughput on a common optimizer-step axis. Panels show completed games per
training block, search visits per move, live replay occupancy, and trainer throughput. Replay capacity
increases in steps, while the small-to-medium transition coincides with lower trainer throughput.

The larger progression across the plotted campaigns reflects successive changes to architecture, data generation,
and training efficiency.

## Training volume and model transition

Table \ref{tab:06-final-chess-recipe-1} summarizes the scale of the final run.

| Quantity | Final training |
| --- | ---: |
| Duration | 2.5 days |
| Model parameters | 6.32 million |
| Completed games | 3.25 million |
| Mean game length | ~104 plies |
| Estimated search simulations | ~100 billion |
| Materialized positions | 209 million |
| Training presentations | 837 million |
| Optimizer steps | 409 thousand |
| Live replay at selection | 16.0 million positions |
| Node rental cost | $43.2 |

We estimate approximately 100 billion search simulations, using 209 million materialized positions and an assumed
average of 500 fresh simulations per position, allowing for subtree reuse and the lower initial visit budgets.
The multiplier is an accounting assumption, not an independently measured average: 209,153,744 × 500 gives
104,576,872,000 simulations before rounding.

Figure \ref{fig:final-training-volume-and-throughput-paper} relates this volume to replay growth and the transition from the 12×128 to the 14×160 model.
The smaller network supported faster early training; median trainer throughput fell from about 17.0 thousand to
11.2 thousand samples/s across the two stages. Search increased from 300 to 600 visits per move over the plotted
run. The configured 800-visit stage lies beyond the selected checkpoint.

![Final model playing strength across measured search budgets](figures/final-search-curve-paper.svg)

Figure: Final playing strength across search budgets. Connected points use the opponent rung with score nearest
50%; pale diamonds show the second opponent-based estimate and bars indicate 95% match-bootstrap intervals.
The horizontal axis lists search budgets categorically.

The expanding replay window retained a broader history of self-play while the learner received approximately four
presentations per admitted position. Throughput, search budget, and replay growth therefore changed together as
training progressed. Loss, learning-rate, and gradient diagnostics are provided in Appendix \ref{app:A}.

The reported checkpoint was selected near the strongest region of the training ladder. A function-preserving
19×176 continuation recovered parity but did not establish a higher plateau.
