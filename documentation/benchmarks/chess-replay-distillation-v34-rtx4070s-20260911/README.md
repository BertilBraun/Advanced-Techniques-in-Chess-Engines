# v34 replay compression into a 0.47M chess network

A student with **474,069 inference parameters** retained a substantial part of the playing strength of
the 6,256,365-parameter v34 generation-1465 teacher. The student is 13.20x smaller by parameter count.
At 64 searches each it trailed by 291 Elo, while a saturated, measured equal-time comparison reduced
the gap to 166 Elo. At equal network MACs it reached statistical parity, but that comparison gives the
student 850 searches and ignores tree-search and launch costs.

This is a successful compression result. It does not establish the student's absolute Elo or its
strength at 10,000--80,000 searches. Superhuman strength at those budgets is plausible, but remains an
untested hypothesis; the equal-search gap may widen as search deepens, as it did in the earlier
distillation probe.

## Provenance

| | |
| --- | --- |
| Training source revision | `92c9f487d38467ac8c34bb88a28604e3a539e15f` |
| Saturated-throughput source revision | `245d71725643884fcc9737f7bfd21d9ac3ff5068` |
| Training node | Vast.ai `38.49.42.120:53893`, 8x RTX 4070 SUPER |
| Evaluation node | Vast.ai `98.142.241.120:23421`, 1x RTX 4070 SUPER 12 GiB, driver 580.95.05, 220 W |
| Evaluation runtime | Python 3.12.3, torch 2.12.1+cu126, CUDA runtime 12.8.93 |
| Teacher | v34 generation 1465, 14x160, 6,256,365 inference parameters |
| Teacher checkpoint SHA-256 | `402efb61146b5a0f569c960e1f7a7e7714ba1bb28982fcdfb7f10e8ec5a98ad6` |
| Student | 8x56, seed 20260827, 474,069 inference parameters |
| Student checkpoint SHA-256 | `d41ba5a39316cbe0acfa24533f83c3a5ae96f3241231f1dc7cb0429afb0aa8c7` |
| Production configuration SHA-256 | `5a1194975e225d4c5ebc329ca7641b205e7cd17b88d5b4aa24679d9e6a939c9a` |
| Saturated-match configuration SHA-256 | `c8ac0bbc37c30d81dde0f241724a21ba58f2763f2b9730a599a4d71bcea32a9f` |
| Replay SHA-256 | `d668a18e07be099538a79a70b2c65c275a724b7dba56eddeb5629d0a7df52a83` |
| Openings SHA-256 | `40582c4f753e90d8ebd170498c37e3f4812bff38287e6bfa6db376e9eec70d39` |

The complete fetched evidence remains under
`.codex-diagnostics/v34-replay-distillation-complete-20260911/`,
`.codex-diagnostics/v34-distill-mac-evaluation/`, and
`.codex-diagnostics/v34-distillation-throughput-and-equal-time-20260911/`. Compact raw artifacts are
committed beside this note, with hashes in `SHA256SUMS`.

| Fetched archive | SHA-256 |
| --- | --- |
| Replay training and initial evaluation | `959bd32cd1225033dd8e272fa5d5050eb6ad67d0ae6470a0296b7c23f1311ef2` |
| Equal network MAC match | `ee8240ee2dcea99fda2f1f7617cf1fc2180c971976048fa7db2397ed1d6a5752` |
| Saturated throughput and equal-time match | `f6abf56a056977b87d30dd24b419870305741cf202e668691bed45a2b50788ac` |

## What was distilled

This experiment compressed a frozen production replay store. It is replay-target compression rather
than teacher-logit distillation. Each record retains the encoded state, legal actions, sparse MCTS visit
counts, discounted outcome WDL, root value, sample weight, policy surprise, source model generation, and
timestamp. The replay does not retain the teacher's raw policy logits or raw network WDL prediction.

The store held 10,000,000 records when frozen. Over v34, 187,612,699 records had been appended and
177,612,699 evicted, so the student saw only the final retained slice. The 10 million records are an
upper bound on distinct board states because duplicates were not deduplicated. Every arm consumed
102.4 million sample presentations: 100,000 optimizer steps at batch 1,024, nominally 10.24
presentations per retained record.

That data exposure is much smaller than the teacher's. The teacher learned online across the full run,
including positions already evicted from the final replay window. This experiment therefore measures
how much strength can be recovered from one frozen 10-million-record slice, not the maximum capacity of
a 0.47M-parameter network.

## Training and selection

Four approximately 0.5M-parameter backbones were trained twice. All arms used AdamW at 0.002, batch
1,024, 1,000 warmup steps, a held learning rate followed by cosine annealing over the final 20%, a 2%
holdout, and 100,000 steps. Selection used the mean held-out policy cross-entropy gap above the target
entropy floor.

| Architecture | Approximate parameters | Seed gaps | Mean policy gap |
| --- | ---: | --- | ---: |
| 4x80 | 496,063 | 0.5365, 0.5367 | 0.5366 |
| 5x72 | 502,243 | 0.5272, 0.5284 | 0.5278 |
| 6x64 | 470,295 | 0.5294, 0.5281 | 0.5288 |
| **8x56** | **475,023** | **0.5194, 0.5248** | **0.5221** |

The selected seed, 20260827, finished at held-out policy loss 1.8393 and WDL loss 0.7041. Its policy
gap was 0.5194. The exact deployed TorchScript network contains 474,069 parameters; the planning count
above includes a small difference in exported versus training structure.

## Match results

Elo is student minus teacher. Confidence intervals are 95% paired bootstraps over opening pairs. All
matches used balanced openings, `parallel_searches: 1`, and terminated all games naturally.

| Comparison | Games | Teacher/student searches | Student W/D/L | Score | Elo difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal search | 200 | 64 / 64 | 19 / 25 / 156 | 0.1575 | -291.3 [-346.1, -243.6] |
| Saturated equal expected time | 400 | 64 / 186 | 63 / 96 / 241 | 0.2775 | **-166.2 [-200.2, -136.0]** |
| Equal network MACs | 200 | 64 / 850 | 78 / 64 / 58 | 0.5500 | +34.9 [-6.9, +77.7] |

These answer different questions:

- Equal search measures how much playing strength the smaller function retained per MCTS search.
- Equal expected time is the practical deployment comparison on this engine, GPU, root population,
  and batching regime. It gives the student 2.907x as many searches.
- Equal network MACs gives 13.28125x as many searches because the student's network forward pass has
  13.28125x fewer multiply-accumulates. It excludes CPU tree work, kernel launches, and imperfect GPU
  utilisation, so it is an arithmetic upper bound rather than an equal-wall-clock result.

The equal-time JSON is labelled `mode: equal-compute` because that was the existing CLI enum used to
pin an externally measured ratio. Its 2.90698 ratio and 64/186 budgets are the authoritative meaning.

## Saturated throughput measurement

Short fixed-root probes tested inference batch caps from 16 through 256. The root population was kept
active during measurement so long-tail finishing games could not shrink and reuse batches. Batch 96
gave the largest repeatable student advantage.

| Batch cap | Repeats | Median student/teacher search throughput |
| ---: | ---: | ---: |
| 16 | 3 | 1.930x |
| 32 | 3 | 2.635x |
| 64 | 3 | 2.450x |
| **96** | **3** | **2.961x** |
| 128 | 1 | 2.150x |
| 256 | 1 | 1.989x |

Two longer 120-second-per-network probes at batch 96 measured 2.9726x and 2.8413x. Their midpoint,
2.90698x, fixed the match budget at 186 student searches against 64 teacher searches. The teacher ran
at about 40,921 searches/s and the student at about 118,955 searches/s across those two probes.

An earlier underfilled measurement used only 100 roots and produced a 1.6167x ratio and a 64/103 match
at -269 Elo. It is superseded by the fixed-root saturated measurement; the difference demonstrates that
an equal-time claim is specific to the serving workload, not an intrinsic property of the checkpoint.

## Interpretation and limits

The practical result is that reducing the network by 13.20x costs about 166 Elo when each side receives
the number of searches it can sustain under the measured saturated batch-96 workload. That is a much
smaller loss than the 291 Elo equal-search result and confirms that the compact model's throughput is
useful. The gap remains decisive: the student scored 27.75% even with 2.91x the searches.

The equal-MAC result shows that enough extra search can erase the shallow-search gap, but the engine
cannot realize 13.28x more searches in equal wall time on this hardware. It should not be presented as
a practical speed comparison.

Absolute student Elo was not measured against Stockfish, and none of these matches used 10,000 or
80,000 searches. Search can magnify policy differences, so extrapolating the shallow result to those
budgets would be unreliable. A high-search Stockfish match would be required before calling the student
superhuman as a measured result.

Only one teacher checkpoint and one selected student seed received game evaluation. Architecture
selection used two seeds, but the match interval does not include training-seed variance. Throughput is
specific to one RTX 4070 SUPER, batch 96, 200 active roots, and this search implementation.

## Conclusion

The first replay-compression experiment is complete. A roughly 0.5M-parameter network recovered enough
of v34 to trail by 166 Elo in the most relevant measured equal-time setup despite training on at most 10
million distinct replay records. The useful follow-up, if pursued, is controlled label reconstruction on
the same positions using raw teacher policy and WDL outputs, followed by the same shallow evaluation.
High-search evaluation is a separate question and is not required to close this phase.
