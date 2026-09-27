# 8. Playing strength and model compression

The final model reached **3,251 benchmark Elo at 100,000 searches per move**, estimated at under five seconds
of thinking time on an RTX 4070 SUPER. This chapter examines the strength gained from search
and the performance retained by a much smaller distilled model.
Table \ref{tab:02-methodology-and-evidence-1} contains the full paired-match results; Appendix \ref{app:B} specifies the evaluation protocol.

## Strength across search budgets

Search increased playing strength from **1,658 Elo for policy-only play** to **3,251 Elo** at the
deepest budget, a gain of 1,593 points. Figure \ref{fig:final-search-curve-paper} shows continuing improvement with diminishing returns:
the final tenfold increase in search added 137 Elo, substantially less than the increases at shallow budgets.

The two opponent-based estimates at the deepest budget differ by only four Elo. At shallower budgets the estimates
are farther apart, as shown by the paired points. Appendix \ref{app:B} describes the calibration and rating calculation.

Deeper searches use additional leaf parallelism to improve GPU utilization. The evaluation curve uses one leaf
at a time at 100 and 1,000 searches, four at 10,000, and sixteen at 100,000.
Section \ref{sec:04a-search} examines the strength–latency tradeoff.

## Distilling a smaller player

The distilled student has approximately 470 thousand parameters, **13.4 times fewer** than the teacher.
It trained for 110,000 optimizer steps on a frozen 20-million-position replay snapshot.
Increasing search from 10,000 to 100,000 raised its rating from 2,697 to 2,873 Elo (Table \ref{tab:07-final-run-results-1}).

| Searches | Elo | 95% interval |
| ---: | ---: | ---: |
| 10,000 | 2,697 | 2,640–2,753 |
| 100,000 | 2,873 | 2,819–2,935 |

The small network retains substantial playing strength, but the gap to the teacher remains despite deeper search.
