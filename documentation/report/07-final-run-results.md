# 8. Playing strength and model compression

The final model reached approximately **3,250 benchmark Elo at 100,000 searches per move**, corresponding to
roughly five seconds of thinking time on an RTX 4070 SUPER. This chapter examines the strength gained from search
and the performance retained by a much smaller distilled model.
Table 1 contains the full paired-match results; Appendix B specifies the evaluation protocol.

## Strength across search budgets

Search increased playing strength from approximately **1,660 Elo for policy-only play** to **3,250 Elo** at the
deepest budget, a gain of about 1,590 points. Figure 13 shows continuing improvement with diminishing returns:
the final tenfold increase in search added roughly 140 Elo, substantially less than the increases at shallow budgets.

![Final model playing strength across measured search budgets](figures/final-search-curve-paper.svg)

Figure 13: Final playing strength across search budgets. Connected points use the opponent rung with score nearest
50%; pale diamonds show the second opponent-based estimate and bars indicate 95% match-bootstrap intervals.
The horizontal axis lists search budgets categorically.

The two opponent-based estimates at the deepest budget differ by only four Elo. At shallower budgets the estimates
are farther apart, as shown by the paired points. Appendix B describes the calibration and rating calculation.

Deeper searches use additional leaf parallelism to improve GPU utilization. The evaluation curve uses one leaf
at a time at 100 and 1,000 searches, four at 10,000, and sixteen at 100,000.
[Section 4.1](04a-search.md) examines the strength–latency tradeoff.

## Distilling a smaller player

The distilled student has approximately 470 thousand parameters, **13.4 times fewer** than the teacher.
It trained for 110,000 optimizer steps on a frozen 20-million-position replay snapshot.
Increasing search from 10,000 to 100,000 raised its rating from approximately 2,700 to 2,870 Elo (Table 5).

| Searches | Elo | 95% interval |
| ---: | ---: | ---: |
| 10,000 | 2,700 | 2,640–2,750 |
| 100,000 | 2,870 | 2,820–2,940 |

The small network retains substantial playing strength, but the gap to the teacher remains despite deeper search.
