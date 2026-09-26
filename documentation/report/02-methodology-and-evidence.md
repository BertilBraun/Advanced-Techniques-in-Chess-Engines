# 2. Evaluating the chess system

We measure thinking effort in search visits rather than seconds. For a given model and search configuration, this
budget is hardware-independent: faster hardware finishes sooner instead of receiving more search. Fixed budgets
also keep background load from changing the amount of search performed. We report approximate thinking time
separately to give these budgets a practical scale.

The most direct test of the system is whether its final model wins games. We played it against Stockfish 13 at fixed
search limits, starting from 50 openings and playing each once with each colour. The model played without search and
at four progressively larger search budgets. For each budget, we tested two Stockfish limits rather than trusting a
single opponent.

## Final playing strength

Table 1 gives all ten matches. Each row contains 100 games; wins, draws, and losses are from our model's perspective.
The bold rating for each model budget comes from the opponent against which it scored closest to 50%, where the
rating estimate needs the least extrapolation.

| Model searches | Parallel | Stockfish nodes | W/D/L | Score | Benchmark Elo (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Policy only | -- | 1,000 | 32/24/44 | 0.440 | **1,658 [1,608, 1,710]** |
| Policy only | -- | 2,000 | 16/22/62 | 0.270 | 1,717 [1,638, 1,790] |
| 100 | 1 | 5,000 | 51/28/21 | 0.650 | 2,328 [2,276, 2,384] |
| 100 | 1 | 10,000 | 39/18/43 | 0.480 | **2,456 [2,400, 2,512]** |
| 1,000 | 1 | 20,000 | 47/40/13 | 0.670 | 2,823 [2,774, 2,873] |
| 1,000 | 1 | 50,000 | 21/48/31 | 0.450 | **2,925 [2,875, 2,977]** |
| 10,000 | 4 | 50,000 | 45/40/15 | 0.650 | 3,068 [3,023, 3,120] |
| 10,000 | 4 | 100,000 | 30/44/26 | 0.520 | **3,114 [3,065, 3,163]** |
| 100,000 | 16 | 100,000 | 51/38/11 | 0.700 | 3,247 [3,192, 3,305] |
| 100,000 | 16 | 200,000 | 25/56/19 | 0.530 | **3,251 [3,206, 3,297]** |

At the deepest budget, the model scored 3,251 benchmark Elo against the harder opponent and 3,247 against the other:
the two measurements agree closely. The ratings use a published calibration of Stockfish's fixed node limits [9].
Chapter 8 examines how strength changes with search; Appendix B gives the rating calculation and interval method.

## Reading the component experiments

The rest of the paper asks why the system reached that result. A faster inference engine can supply more search,
but that only helps training if the system finishes more games and admits useful positions to replay. A lower loss
on stored positions can indicate a better fit, but the real test is whether the next model plays better. We follow
each proposed improvement as far through this chain as the experiment measured.

For online comparisons, we matched starting weights, replay, evaluation, and hardware where possible. Smaller
frozen-data tests helped screen ideas before expensive self-play runs. Because the final recipe combines many
changes, we attribute a separate strength gain to a component only when a comparison actually measured one.
