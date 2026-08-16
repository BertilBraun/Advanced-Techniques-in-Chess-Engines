# Generation 445 search scaling versus the Stockfish 13 fixed-node ladder

## Result

Generation 445 scored 49 wins, 34 draws, and 17 losses in 100 games against Stockfish 13 limited to 6,500 nodes
per move. The candidate score was 66.0%, with a paired-bootstrap 95% interval of 58.5% to 73.5%. The match used
50 opening pairs sampled from the frozen 200-position balanced elite suite, with both colors played from every
position.

Under the logistic Elo transform, the score corresponds to a descriptive difference of +115 Elo, with the score
interval transforming to +60 to +177 Elo. The published Stockfish 13 fixed-node curve places 6,500 nodes at
approximately 2,300 Elo. Conditional on that approximate anchor, this gives a point estimate near 2,415 and an
interval of approximately 2,360 to 2,477. The appropriate rounded summary is therefore **about 2,400 calibrated
Elo, with a match-sampling range of roughly 2,360 to 2,480**.

This is an Elo-like interpolation on the cited engine calibration, not a FIDE, online-platform, or universal engine
rating. The interval includes match sampling uncertainty but not uncertainty in reading or transferring the external
Stockfish calibration.

## Exploratory ladder

Every rung used generation 445, exactly 1,000 candidate searches per move, one parallel search, the same five
color-swapped opening pairs, and the same match seed. Elo labels are approximate readings from Marco Meloni's
[Stockfish 13 fixed-node curve](https://www.melonimarco.it/2021/03/02/stockfish-13-e-lc0-test-al-variare-del-numero-di-nodi/).

| Approximate calibrated Elo | Stockfish 13 nodes | Wins | Draws | Losses | Score | Paired 95% interval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,700 | 1,200 | 9 | 1 | 0 | 95% | 85%–100% |
| 1,900 | 2,100 | 9 | 0 | 1 | 90% | 70%–100% |
| 2,100 | 3,500 | 10 | 0 | 0 | 100% | 100%–100% |
| 2,300 | 6,500 | 4 | 4 | 2 | 60% | 30%–75% |
| 2,500 | 11,000 | 2 | 3 | 5 | 35% | 15%–50% |
| 2,700 | 20,000 | 1 | 4 | 5 | 30% | 25%–40% |
| 2,900 | 40,000 | 1 | 1 | 8 | 15% | 0%–35% |
| 3,100 | 100,000 | 0 | 2 | 8 | 10% | 0%–20% |

The 6,500-node rung was selected for the 100-game match because its 60% probe score was closest to 50%. The larger
match's 66% score shows that the ten-game probe was directionally useful but too noisy to identify the exact crossing.
The 6,500- and 11,000-node results jointly place the crossing near 2,400 on this calibration.

## Direct 64-search comparison

The 64-search ladder repeated the same generation, Stockfish rungs, five opening pairs, color swaps, and random
seeds. Only the candidate search budget changed from 1,000 to 64.

| Approximate calibrated Elo | Stockfish 13 nodes | Wins | Draws | Losses | Score | Paired 95% interval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,700 | 1,200 | 8 | 1 | 1 | 85% | 65%–100% |
| 1,900 | 2,100 | 1 | 6 | 3 | 40% | 25%–60% |
| 2,100 | 3,500 | 2 | 3 | 5 | 35% | 15%–60% |
| 2,300 | 6,500 | 1 | 2 | 7 | 20% | 0%–50% |
| 2,500 | 11,000 | 0 | 2 | 8 | 10% | 0%–20% |
| 2,700 | 20,000 | 0 | 1 | 9 | 5% | 0%–15% |
| 2,900 | 40,000 | 0 | 0 | 10 | 0% | 0%–0% |
| 3,100 | 100,000 | 0 | 0 | 10 | 0% | 0%–0% |

A one-parameter logistic fit to the eight rung scores gives approximately 1,972 calibrated Elo at 64 searches. The
same fit to the matched 1,000-search probes gives approximately 2,471 Elo; the independent 100-game result gives
approximately 2,415. The observed 64-to-1,000-search gain is therefore roughly **440 to 500 Elo** for generation
445. This is a 15.625-fold budget increase, or 3.97 doublings, corresponding to an average of about 112 to 126 Elo
per search doubling over the measured range.

The fit is descriptive. It treats a draw as half a point and does not model the correlation from reusing five
opening pairs at every rung. The paired intervals in the tables convey how noisy each ten-game probe remains.

## Reproducibility

- Evaluation source revision: `ca942780260b6c0744a80aaa8c4333c52e7f9bf1`
- Candidate inference model SHA-256: `0454710e39e2e61fe398a8812b7e97599d32011977d03197bc077327c52d66bd`
- Stockfish identity: Stockfish 13
- Stockfish executable SHA-256: `7f7c4a7ec7362eecfef72d1be2ade0592693ef60b7eee3fc6505db0fc479713e`
- Opening manifest SHA-256: `61115e8c8e7eed8cc7125b249e3810418e9cf6be7cd32771986b06d68d9c072b`
- Final result SHA-256: `75745fe98b3938b84a42abf76f0191ae64f027468f54f5d1b91e498b403721ec`
- Final opening-selection seed: `20260817`
- Final match seed: `20260819`
- Candidate budget: 1,000 searches, one parallel search, one inference worker, batch size 64, one outstanding batch
- Final wall time: 1,563.6 seconds on eight RTX 3060 GPUs while production training remained active
- 64-search ladder result SHA-256 values, in ascending node order:
  `1584c523d6bf66cd515280a9acafdeb8c315b1c1a51177e1041561d6d4f2f856`,
  `372af64ad5210fe8ac1b6cc1a1dc4f00216564691bb26cd49cc641796c6e1481`,
  `1f3f3779652722cf056d592ba72af91b30ad22e368128d29f8c15e908cec7f5a`,
  `fa75a50f31eca9dd4ee8061bcfa0ed098c5816f02b33e04ca2ec8bddc1dba8a5`,
  `437ff7f73b0e26006300896a204b4c32b33e62b89d19082025a58d4367686a5c`,
  `3433dd6e85d7f34b23f58a6453860224aebf03d40e6ca9c0b6008ef79c544ae8`,
  `4c14046fa2e8fdd7f543d16f2420ab4aea65663feb462a1abe976912dcd635dd`, and
  `c4ba7cbdf90dfec965c10ae1d90c0a51c10fe6025025a5f3736b72d74bf56899`

The complete result and all shard files were copied from the ephemeral compute node to
`.codex-diagnostics/chess-evaluation-g445-s1000/` on the development host. Production training was not stopped and
advanced from generation 446 to generation 455 during the 1,000-search evaluation. The 64-search ladder was copied
to the same evidence directory after its completion.

## Interpretation limits

The direct ladders isolate search budget much better than the scheduled Stockfish 18 evaluation, but they still use
only five opening pairs per rung. The measured 112-to-126-Elo gain per doubling should not be extrapolated unchanged:
search returns normally diminish. Extending the lower end of this range through the 5.13 doublings from 1,000 to
35,000 searches would add about 575 Elo, but this is an optimistic constant-slope extrapolation rather than a result.
A timed, empty-GPU ladder is required to measure the actual deployment strength.
