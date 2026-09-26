# 8. Final training and evaluation results

The 6.32-million-parameter final model reached **3,251 benchmark Elo at 100,000 searches per move** after
2.5 days of training. This chapter relates that result to the training trajectory, the gain from additional
search, and the strength retained by a compact distilled student. All final ratings use the paired Stockfish 13
evaluation introduced in Chapter 2.

## Training progress

The 64-search training ladder rose from 798 to 2,372.2 over 2.5 days, reaching a peak of 2,407.6 within that
window. Figure 11 compares the trajectory with four earlier chess campaigns. The final recipe reached a higher
plateau within its shorter training window; [Appendix B](appendix-b-evaluation-tables.md) specifies the
comparison windows and estimator adjustment.

![64-search ladder Elo across five chess training campaigns](figures/chess-ladder-progress-paper.svg)

Figure 11: Smoothed 64-search training ladders across five chess campaigns. The final curve ends at 2.5 days and
the preceding baseline at 3.0 days; the matched comparison uses a common rating calculation.

Using a common three-rung estimator, the preceding baseline and final recipe have plateau ratings of 2,283.9
and 2,358.0, respectively: an improvement of approximately 74 Elo. This comparison summarizes the change in the
complete training recipe. The estimator transfer and its sensitivity are detailed in Appendix B.

Checkpoint 1026 was selected near the strongest region of the 64-search ladder. The reported architecture has
14 residual blocks and 160 channels; a function-preserving 19×176 continuation recovered parity without
establishing a higher plateau. Appendix A identifies the training sequence underlying the reported checkpoint.

Training comprised **408,500 optimizer steps** in 817 quanta, or **836,608,000 presentations** at a global batch
of 2,048. Self-play completed **3,249,647 games** and materialized approximately **209.15 million net positions**,
corresponding to roughly four training presentations per admitted position. The replay window contained
**16 million live rows** at selection. Appendix A provides the volume and throughput traces.

Using a conservative mean of 100 searched plies per game and roughly 600 simulations per move gives an estimated
195 billion search simulations across the 3.25 million completed games.

Median trainer throughput was 16,977 samples/s over 480 small-model quanta and 11,194 over 337 medium-model
quanta. The 2.5-day run cost **$43.20** at the node rental rate of $0.72/h.

## Playing strength across search budgets

Search increased playing strength from **1,658 benchmark Elo for policy-only play** to **3,251 at 100,000
searches per move**, a difference of 1,593 points. Figure 12 summarizes the ten matches in Table 1, showing both
opponent-based estimates at each budget. Strength continued to improve through the deepest measured point, with
diminishing Elo gains for successive increases in search.

![Final model playing strength across measured search budgets](figures/final-search-curve-paper.svg)

Figure 12: Connected points use the opponent rung with score nearest 50%; pale diamonds mark the other tested rung,
and bars show 95% match-bootstrap intervals. The categorical horizontal axis separates policy-only play from
100, 1,000, 10,000, and 100,000 searches per move; its spacing does not represent compute.

At the deepest budget, the 100,000-node and 200,000-node Stockfish opponents imply 3,247 and 3,251 benchmark
Elo, respectively, placing both estimates close to the reported 3,251. A 100,000-search move takes roughly
five seconds of thinking time on an RTX 4070 SUPER with the current setup.

### Parallel search trades time for strength

The evaluation uses increased leaf parallelism at deeper budgets to reduce latency. As discussed in Section 4.1,
concurrent leaf selection trades some search quality for better GPU utilization. The strength curve uses one parallel search at
100 and 1,000 searches per move, four at 10,000, and sixteen at 100,000. A controlled 1,000-search sweep against
the same 20,000-node opponent illustrates the tradeoff: one parallel
search measured 2,823 Elo, four measured 2,804, and sixteen measured 2,778. Their central differences of 19 and
45 Elo are smaller than the overlapping match intervals, while match time fell from about **18.3 minutes** to
**3.6** and **1.6 minutes**, respectively. Four-way and sixteen-way parallelism made those matches roughly five
and eleven times faster.

At only 100 searches, sixteen-way parallelism reduced the central strength estimate by 235 Elo. The available
points indicate a smaller penalty at larger budgets, but do not establish a general concurrency schedule.
[Appendix C](appendix-c-supporting-comparisons.md) gives the controlled operating points.

## Distilling a smaller player

A 470,295-parameter student—13.4 times smaller than the teacher—reached **2,697 [2,640, 2,753] benchmark Elo**
at 10,000 searches after 110,000 training steps. The shorter 36,621-step student reached **2,683 [2,637, 2,731]**
under the same opponent and search setting. The 14-Elo difference is unresolved by the match intervals, consistent
with the nearly flat held-out policy loss. At 100,000 searches, the longer student reached
**2,873 [2,819, 2,935]** against the tested opponent.

Both students trained on the same frozen 20-million-row replay snapshot. The longer schedule therefore increased
exposure to fixed data rather than adding new self-play experience, with no resolved strength gain at the shared
10,000-search budget. Appendix B gives the match counts, intervals, and evaluation settings.
