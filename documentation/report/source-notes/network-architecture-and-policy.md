# Network architecture and policy source dossier

This is a source dossier, not report prose. It is organized by technical question rather than by run chronology.
Internal run identifiers are intentionally omitted. Commit links are included only where the repository has no
equivalent preserved benchmark artifact. Numerical claims must retain the caveats recorded here when they are turned
into publication text.

## What constitutes a policy representation

### Question

- Which concepts must be kept separate when comparing policy designs?

### Approaches and mechanisms

- The **action encoding** assigns an integer ID to each move. The current chess contract has 1,880 canonical actions.
- The **network output layout** determines what logits the model emits before the action map is applied. This project
  used both an action-sized vector and a 76-by-64 policy-plane tensor.
- The **policy head** maps trunk features into that output layout. Dense, spatial-plane, and from-to attention heads
  are different heads even when they ultimately supervise the same 1,880 canonical actions.
- The **legal-action mask** determines which output logits enter a position's softmax. It is not part of the learned
  head. The 4,864-output plane representation relied on masking impossible and illegal slots downstream.
- The **policy target** is the search-visit distribution over legal canonical actions. Changing the head or output
  layout does not by itself change the target-generation algorithm.

### Decision rationale

- The report must not call a change from 1,880 to 4,864 outputs merely a “new head.” That change also altered the
  Python/native action ABI, symmetry permutation, replay dimensions, evaluation metric, and checkpoint compatibility.
- Conversely, dense-to-from-to comparisons on the same 1,880-action encoding are genuine head comparisons when the
  trunk and data are held fixed.

### Pitfalls and unresolved evidence

- An obsolete history note says the action vector contained 1,814 entries. The executable contract and preserved
  modern evidence say 1,880; the old number must not be repeated without reconstructing its exact historical schema.
- Cross-entropy values from an unmasked 1,880-way softmax and a legal-masked policy loss are not numerically
  comparable. The regression analysis explicitly warns about this.

### Sources

- [Current chess action contract](../../../py/src/games/chess/contract.py)
- [Current from-to action table](../../../py/src/games/chess/policy_encoding.py)
- [Native policy-encoding authority](../../../cpp/src/games/chess/encoding/ChessPolicyEncoding.cpp)
- [Regression analysis of the representation and loss changes](../../plan/chess-post-four-day-regression-analysis-20260820.md)

## Chess input representation, history, and symmetry

### Question

- What information reaches every trunk, and how are colour and augmentation symmetries made consistent with policy
  and value targets?

### Representation

- The current network input is 52 planes over an 8-by-8 board. The replay representation packs 40 binary planes as
  bitboards and 12 scalar values as signed bytes; batch construction expands every scalar across its spatial plane.
- The 40 binary planes, in contract order, are:
  - six own piece-type planes and six opponent piece-type planes, ordered pawn, knight, bishop, rook, queen, king;
  - own kingside and queenside castling rights, then opponent kingside and queenside castling rights, each broadcast
    over the board;
  - aggregate own occupancy and aggregate opponent occupancy;
  - current checking pieces;
  - the en-passant target square;
  - broadcast indicators that the current position occurred at least once and at least twice previously;
  - origin and destination planes for each of the eight most recent moves, newest-first in the native board history;
  - a fixed checkerboard-colour plane;
  - a broadcast opposite-coloured-bishops indicator, true only when exactly one bishop remains for each side and
    they occupy opposite square colours.
- The 12 scalar planes are:
  - own-minus-opponent material counts for pawn, knight, bishop, rook, queen, and king;
  - the fifty-move counter, capped at 100;
  - the side-to-move player's counts of pawns, knights, bishops, rooks, and queens.
- Repetition history is exact within the native board's bounded reversible history, which retains at most 100 plies
  and resets on a pawn move, capture, castling-right change, or capacity boundary. Recent-move planes retain eight
  moves independently of the two repetition-threshold planes.

### Canonicalization and augmentation

- Native encoding is always from the side-to-move perspective. If Black is to move, ranks are flipped and the
  colour sources are swapped, so the first six piece planes and first two castling planes still describe “own.”
  Side to move therefore does not need a separate scalar plane.
- This colour canonicalization is not the training augmentation. The only chess augmentation choices are identity
  and a file mirror.
- File mirroring reverses board columns, swaps own kingside with own queenside castling and does the same for the
  opponent, and restores the checkerboard plane to its canonical fixed orientation. Scalar planes are unchanged.
- The corresponding native action permutation mirrors every canonical action ID. Batch construction applies that
  same permutation to the sparse primary policy, its legal-action set, and policy-shaped auxiliary targets. WDL and
  scalar auxiliary targets do not change under a file reflection.

### Evidence and decision rationale

- The input/action flip harness compared a position with its colour-mirrored counterpart, checked the native policy
  mapping, and verified file-mirror behavior. Dedicated Python tests cover castling-plane swaps, checkerboard
  restoration, native mirrored encodings, and the action permutation's involution.
- The history expansion from the older 29-plane contract to 52 planes added explicit recent moves and rule-sensitive
  state. It is retained because repetition, castling, en passant, and reversible move history affect legal search and
  draw semantics; omitting them makes positions that require different decisions indistinguishable.
- This is a representation/correctness decision. The repository does not contain a one-variable Elo ablation for
  the full 52-plane representation against the older input.

### Pitfalls and unresolved evidence

- A FEN alone does not reconstruct earlier repetition occurrences or the full eight-move history. Tools that build a
  board from FEN without a move sequence cannot claim parity with live self-play inputs.
- Mirroring files without exchanging the kingside/queenside castling planes was a historical augmentation defect.
  The current contract and tests include that swap.
- Parameter and throughput counts measured with the older 29-plane start convolution are not exact counts for the
  current 52-plane model.
- The independent strength contribution of recent-move, repetition, checkerboard, material-count, and
  opposite-coloured-bishop planes has not been isolated.

### Sources

- [Native 52-plane encoder](../../../cpp/src/games/chess/encoding/ChessEncoding.cpp)
- [Native representation dimensions](../../../cpp/src/games/chess/encoding/ChessEncoding.hpp)
- [Bounded board history](../../../cpp/src/games/chess/implementation/ChessBoard.hpp)
- [Python representation and augmentation contract](../../../py/src/games/chess/contract.py)
- [Mirror-augmentation regression tests](../../../py/test/test_chess_mirror_augmentation.py)
- [Native/Python flip harness](../../../cpp/test/flip-harness/run_checks.py)
- [History-expansion implementation](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/e98d670c)

## Dense reduced-action policy heads

### Question

- Can a compact convolutional trunk directly predict the canonical action vector, and how much capacity should the
  final projection receive?

### Approaches and mechanisms

- The original chess head used `1x1 convolution -> batch normalization -> ReLU -> flatten -> linear`, with the final
  linear layer producing one logit per canonical action.
- Dense configurations with two, four, and eight projected spatial channels appear in the implementation and
  experiment history. Channel count changes the flattened field and therefore the cost of the final action-sized
  projection; it is a capacity choice within the same representation, not a different action encoding.
- The unbottlenecked four-channel form had about 483,680 policy-head parameters on the controlled 12-by-128 trunk.
  The next-policy auxiliary used another action-sized head, so dense policy projections could occupy a substantial
  fraction of a small model.
- Seven dense variants were implemented in the supervised bake-off:
  - four channels with no reduction;
  - four channels followed by one unpadded 3-by-3 spatial reduction;
  - eight channels followed by two spatial reductions;
  - four channels with rank-64 or rank-96 factorization of the final linear projection;
  - eight channels with one spatial reduction plus a rank-64 or rank-96 projection.
- The low-rank form factors the final map into `flattened features -> bottleneck rank -> canonical actions`. It reduces
  parameters without changing the action representation.

### Evidence and results

- The implementation and seeded frozen-replay bake-off are preserved in Git, but the result JSON named by the chosen
  configuration was not committed under `documentation/benchmarks`.
- The owner remembers a broader set of roughly ten policy-head comparisons but does not know of a surviving external
  result bundle. That recollection establishes that the design space was broader than one comparison; it does not
  recover scores or justify a reconstructed table.
- The configuration decision records the rank-96 four-channel form as tying the 483k-parameter baseline while using
  about 207k parameters. This is useful historical evidence, but it is not equivalent to a preserved raw report.
- A later controlled policy study measured the unbottlenecked dense head as a strong but slower-learning baseline.
  Its held-out policy gap was 0.1069 nats on the fixed teacher dataset.

### Decision rationale

- The rank-96 head was retained when the immediate goal was to preserve the proven reduced-action recipe while
  recovering parameters.
- The dense family was later superseded by the from-to head because the latter preserved per-square structure, used
  roughly one ninth as many head parameters in the controlled comparison, and improved held-out policy fit with the
  trunk fixed.
- “Superseded” is the correct conclusion. Dense heads trained strong networks and were not disproved as a class.

### Pitfalls and unresolved evidence

- The raw dense-variant bake-off report has not been recovered. The precise tie, uncertainty, training horizon, and
  per-variant throughput cannot be publication claims until that artifact is found or the bake-off is reproduced.
- A dense head is disproportionately harmful to a square-token attention trunk because it compresses all 64 token
  vectors to a very small spatial channel field before predicting moves. The controlled evidence shows this is an
  interaction, not a universal statement that dense heads are weak.

### Sources

- [Current selectable dense-head implementation](../../../py/src/training/network.py)
- [Dense-variant bake-off implementation commit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/23eb8e2c)
- [Recorded rank-96 selection rationale](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/3c5d01f6)
- [Controlled dense/from-to comparison](../../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)

## Structured 76-plane policy heads

### Question

- Can the policy preserve chess move geometry and remove the large dense projection by predicting spatial move
  planes?

### Approaches and mechanisms

- The plane layout contains 56 sliding-ray planes, eight knight planes, and 12 explicit promotion planes.
- The first structured implementation emitted the plane tensor and gathered the canonical action logits through a
  fixed native-derived index. This kept a reduced 1,880-action external contract while using spatial computation
  inside the head.
- A subsequent direct-plane representation made the 76-by-64 layout itself the action space: 4,864 raw logits.
  Search normalized only legal action IDs, so impossible plane/square cells did not contribute to the legal policy.
- The repaired plane head was not merely a bare output convolution. It used
  `1x1 convolution -> BN -> ReLU -> 3x3 convolution -> BN -> ReLU -> 1x1 plane projection`, with a 64-channel primary
  hidden field, a 32-channel auxiliary field, and a small output initialization.

### Evidence and results

- The native mapping and colour-flip contract were checked over 83,651 moves without a discrepancy. This validates
  the mapping, not the learning quality of the representation.
- Controlled inference showed that the direct-plane ABI did not explain the observed system slowdown: reconstructed
  CNN and attention controls were approximately level with their historical dense-head rates.
- In the short supervised testbed, the repaired plane CNN reached 2.1828 held-out policy cross-entropy after about
  2,500 steps versus 2.0824 for the dense control. The plane curve was still improving faster, the intended budget
  was not completed, and the dense primary plus auxiliary heads carried about 2.4 million additional parameters.
- Failed online attention attempts using the direct-plane system plateaued at roughly 650 ladder Elo, but those
  attempts also had broken generation-zero priors and replay-ingestion starvation. They are incident evidence, not a
  clean policy-representation ablation.
- The owner remembers a plane-policy model—described from memory as a 96-plane head—also training in self-play after
  the early failures, learning more slowly, and underperforming enough to be discarded quickly. The recovered
  implementation and action layout use 76 planes, and no matching result artifact has been identified, so the report
  must preserve both the qualitative recollection and the unresolved plane-count/artifact identity.

### Decision rationale

- The direct 4,864-action ABI was removed to re-establish a trustworthy comparison with the proven 1,880-action
  recipe. The mapping itself was not found incorrect.
- The repaired 76-plane head remained technically viable, but the project did not complete a clean long online
  comparison that isolated it. The later from-to head supplied stronger controlled learning evidence while retaining
  the canonical 1,880-action contract, so the plane family was superseded rather than conclusively rejected.
- Owner recollection explains why it was superseded quickly in practice, but cannot upgrade the missing online
  comparison into a quantitative rejection.

### Pitfalls

- The original direct plane head was a bare `1x1` projection with inappropriate Kaiming-ReLU initialization. On real
  probes, initial policy-logit standard deviation was 8.4-11.9 for attention and 4.6 for the CNN, versus about 1.0 for
  the old dense head. The resulting near-one-hot random prior contaminated initial self-play.
- The attention trunk lacked a final normalization and its linear layers received a convolution-oriented
  initialization. Policy loss and gradient norm diverged rapidly under the inherited high learning rate.
- These failures belong to initialization and optimization as well as head design. They cannot support the claim
  that plane policies or attention trunks intrinsically fail.
- A direct-plane policy changes the evaluation baseline: the unmasked uniform loss is near `log(4864)`, while the
  legal-masked loss is much smaller and position-dependent.

### Unresolved evidence

- No preserved completed head-only comparison establishes whether the repaired plane head catches or beats the dense
  head at convergence.
- No clean online match isolates reduced-action dense versus gathered-plane versus direct-plane representations.

### Sources

- [Structured-head implementation commit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/cbf74199)
- [Direct-plane throughput controls](../../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md)
- [Short supervised plane/dense comparison](../../benchmarks/supervised-testbed-rtx4070-20260821/README.md)
- [Failure diagnosis and measured initialization](../../plan/chess-post-four-day-regression-analysis-20260820.md)
- [Recovery design for the repaired plane head](../../plan/chess-recovery-plan-20260820.md)
- [Restoration of the reduced action contract](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/afefba9a)

## From-to attention policy head

### Question

- Can chess's move structure be expressed directly as relations between origin and destination squares without an
  action-sized dense projection or a mostly empty plane tensor?

### Mechanism

- The head flattens the trunk into 64 square tokens, projects each token through GELU, and creates separate query and
  key vectors.
- A scaled query-key product scores all 64-by-64 origin/destination pairs. A fixed gather maps those scores to the
  1,880 canonical actions.
- A separate projection produces four promotion-piece scores on the last rank. Knight promotion is the reference;
  queen, rook, and bishop offsets are added to the shared origin/destination logit.
- En passant is an ordinary diagonal square pair. The canonical encoder represents castling as king-square to
  rook-square, so it also fits the same table.
- The retained key width is 128. On the controlled convolutional trunk, the head contained 51,072 parameters versus
  483,680 for the dense head.

### Evidence and results

- Holding the 12-by-128 convolutional trunk fixed, replacing the dense head improved the held-out policy gap by
  0.0298 nats, with a 95% paired interval of 0.0285-0.0311 nats.
- Widening the trunk after saving head parameters added only another 0.0018 nats in the measured three-corner
  comparison. The fourth dense-wide corner was not run, so that increment is inferred rather than a full factorial.
- On an attention trunk, dense-to-from-to improved the held-out gap by 0.1573 nats. This much larger interaction is
  consistent with the dense head destroying the per-square representation that the attention trunk builds.
- At the production trunk width, the from-to head cost 1.9% of batch-512 forward throughput and about 9% at batch 64.
  An earlier 21% attribution was wrong because that comparison also widened the trunk into an inefficient channel
  region.
- The native inference path originally cast the head's integer gather buffers to BF16. Indices such as 4,094 are not
  exactly representable, so search failed. Converting only floating-point model state fixed the defect and the real
  native pipeline now has CPU and CUDA-path regression coverage.

### Decision rationale

- Retained because it was the only policy-head change with a controlled, trunk-held-fixed improvement, large
  parameter reduction, and acceptable measured serving cost.
- The evidence is frozen-teacher policy fitting plus throughput, not an isolated long self-play Elo ablation. The
  final system result can support the assembled recipe but cannot assign a standalone Elo gain to this head.

### Pitfalls and unresolved evidence

- The often-quoted conversion from policy cross-entropy to Elo is heuristic and dataset-dependent. The report should
  publish the measured nats and avoid presenting the converted range as match evidence.
- The controlled study ran one seed per cell for 8,000 rather than 80,000 steps. Its bootstrap intervals quantify
  held-out position sampling, not initialization variance.
- The teacher dataset had one side-to-move parity and endgame-conversion defects. They affected all cells, making the
  relative comparison useful but limiting absolute transfer.
- No clean online head-only Elo match exists.

### Sources

- [Current from-to implementation](../../../py/src/training/network.py)
- [Canonical action table](../../../py/src/games/chess/policy_encoding.py)
- [Controlled learning and throughput study](../../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)
- [Native head regression tests](../../../py/test/test_native_new_head_inference.py)

## Auxiliary policy representations

### Question

- Did the project train only the primary search policy, or did it use other policy-shaped heads?

### Approaches and mechanisms

- The retained next-policy auxiliary predicts the later player's search policy at a configured ply offset. It owns a
  separate head but uses the same head architecture and action space as the primary policy.
- The target remains in the future state's own player-to-move canonical action space. Augmentation transforms it in
  that same space; it must not reuse the primary row's legal-action mask.
- Legal-move prediction was implemented as another policy-shaped auxiliary option, but it was not retained in the
  final recipe.
- Auxiliary heads exist only in the training checkpoint. The inference artifact copies the shared trunk, primary
  policy head, and WDL head and deliberately strips training-only heads.

### Evidence and decision rationale

- Executable tests cover distinct next-policy eligibility and masking, symmetry, checkpoint persistence, and removal
  from the inference artifact.
- The next-policy head is retained as cheap supervision already available from the trajectory. There is no matched
  long online ablation isolating its strength contribution.
- Legal-move prediction was not promoted: legality is already known exactly by the game engine, and the repository
  contains no evidence that the extra objective justified its capacity and gradients.

### Pitfalls and unresolved evidence

- Calling next-policy “free” is imprecise. Labels require no extra engine search, but the additional head consumes
  training memory, compute, and shared-trunk gradient capacity.
- The final weight is an assembled-recipe choice, not an isolated optimum.

### Sources

- [Auxiliary-head construction](../../../py/src/training/network.py)
- [Target layouts](../../../py/src/training/targets.py)
- [Runtime architecture and artifact boundary](../../architecture/python-runtime-rework.md)
- [Next-policy masking tests](../../../py/test/test_distillation.py)

## Convolutional trunk family

### Question

- Which trunk provides the best useful learning under the project's small-board inference and self-play constraints?

### Approaches and mechanisms

- The ordinary CNN is an input convolution followed by post-activation residual blocks. Each block is
  `conv -> BN -> ReLU -> conv -> BN -> add skip -> ReLU`.
- Depth and width were varied extensively. Throughput is highly shape- and batch-dependent; equal parameter counts
  do not imply equal serving cost.
- The original progressive controller trained each larger network independently against the same growing replay
  stream. The final capacity investigation also tested a function-preserving transition from the trained 14-by-160
  network into the 19-by-176 network; this is a distinct, manual growth path rather than the controller's ordinary
  candidate initialization.
- The retained shapes avoid a measured inefficient width region and were checked at self-play and evaluation batch
  sizes rather than selected from parameter count alone.

### Evidence and results

- Early contended architecture measurements showed CNN controls processing 15-29% more training samples per second
  and using 1.9-2.4 times less peak allocated memory than matched-parameter attention models.
- In the controlled teacher-data study, a CNN with the same from-to head beat the bare attention trunk by 0.0060
  nats. The gap is modest, but the CNN also had a major memory and serving-throughput advantage.
- Channel width produced a non-monotonic throughput curve. Width 128 was a local optimum; the 132-152 region cost
  materially more than arithmetic predicted. At evaluation batch 64, depth and kernel-launch count dominated width.
- Progressive sizing has a measured throughput premise and a recoverable controller, but the completed campaign also
  exposed two important controller failures: unequal candidate training invalidated loss-based promotion, and a
  from-scratch larger network did not recover the active network's playing strength before promotion.

### Decision rationale

- Retained because the CNN offered the best combined controlled policy fit, inference throughput, training memory,
  mature native deployment, and quantization path.
- The conclusion is workload-specific. It does not establish that transformers are generally inferior for chess.

### Pitfalls and unresolved evidence

- Several architecture measurements were initially confounded by shared GPU load, FP32 instead of production BF16,
  the wrong runtime path, or many widths in one process disturbing autotuning. Only corrected controls should be used.
- Some batch-320 and batch-64 shape results were transcribed before the node was released and lack raw JSON. They can
  explain selection but should not carry a headline quantitative claim without repetition.
- The exact progressive ladder lacks a fixed-size equal-compute comparison.

### Sources

- [Current trunk implementation](../../../py/src/training/network.py)
- [Contended CNN/attention comparison](../../benchmarks/chess-architecture-contended-rtx3060-20260817/README.md)
- [Controlled viability and shape study](../../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)
- [Progressive sizing mechanism](../../architecture/progressive-model-sizing.md)
- [Progressive throughput premise](../../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md)

## Progressive model sizing

### Question

- Can early self-play use a cheaper network for higher data/search throughput, then introduce more capacity only
  after the currently active model's strength curve slows?

### Approaches and mechanisms

- Fixed sizing remains a supported control: one complete model definition trains and serves throughout.
- Progressive sizing owns an ordered set of complete network definitions. The retained sequence is
  12 residual blocks by 128 channels, 14 by 160, and 19 by 176; every stage otherwise shares the same 52-plane input,
  scaled post-activation block family, global context, from-to policy, WDL head, and auxiliary objectives.
- The production controller initializes an ordinary successor independently and trains it from the same captured
  replay description as the active model. A separate end-of-campaign procedure grew a trained 14-by-160 checkpoint
  into 19-by-176 while preserving its function, trained the new capacity, then rebuilt and fine-tuned QAT state.
- Candidate start and candidate promotion are separate gates:
  - **Start gate:** the bias-corrected primary-ladder Elo EMA uses decay 0.90. The slope is computed over the code's
    six-observation window: seven retained EMA samples span six observation-to-observation intervals. Two consecutive
    complete windows strictly below the stage threshold latch the immediate successor. The thresholds are 50
    Elo/hour before the 14-by-160 successor and 4 Elo/hour before the 19-by-176 successor.
  - **Catch-up cadence:** an eligible ordinary successor trains an average of 1.5 optimizer quanta per global
    generation. Since
    a quantum is indivisible, the code alternates one and two quanta from the global generation index. The active
    model always receives one quantum, and replay credit is consumed only once for that active quantum. When two
    candidate quanta run at one boundary, both loaders receive the same global replay-source optimizer step and
    therefore repeat the same deterministic batch sequence while the candidate's optimizer state advances.
  - **Promotion gate:** the current controller schedules a paired candidate-versus-active match at evaluation
    boundaries. A candidate score of at least 0.48 passes; two consecutive completed matches must pass. A failure
    resets the sequence, while a failed or cancelled evaluation supplies no evidence and does not reset it.
- Only the active checkpoint is published to self-play and evaluation. Private candidate checkpoints, optimizer
  progress, match-gate state, the Elo latch, and any partially completed multi-model quantum are persisted for exact
  recovery. Candidate checkpoints referenced by evaluation jobs are pinned against retention.

### Function-preserving growth

- Width growth uses random-in/zero-out wiring: new units compute small nonzero activations, but existing outputs do
  not read them initially. This preserves the parent function while giving every zeroed reader a gradient at the
  first update; zeroing both sides would preserve the function but permanently strand the new capacity.
- Appended residual blocks are identity-initialized by zeroing their final branch output. Existing blocks require
  branch-scale compensation because the configured scale changes with depth. The compensation belongs on the final
  batch-normalization affine parameters, not on its running statistics or the preceding convolution.
- Global-pooling blocks divide channels into local and global groups at one quarter of the width. Widening must map
  channels around that moving boundary or copied units silently change roles.
- The measured 14-by-160 to 19-by-176 growth changed policy logits by at most `1.34e-05`, changed value outputs by
  `1.07e-06`, preserved top-one policy agreement exactly on the probe, and delivered gradients to every zeroed
  reader. A 100-game float match scored 0.455, or -31.4 Elo with a 95% interval spanning parity.
- One float epoch over the live replay window brought the grown network to a 0.495 match score against its parent.
  Direct INT8 conversion nevertheless failed fidelity (`0.759` top-one agreement, `0.102` mean KL, `2.01` maximum
  KL). Ten QAT quanta recovered fidelity to `0.878` / `0.036` / `0.227`; the deployed INT8 network then scored
  0.440, about 42 Elo below the parent estimate. The float match therefore could not decide deployment.

### Evidence and decision rationale

- The throughput premise is measured: small networks can evaluate materially more positions early in the workload.
- The implementation has explicit candidate order, same-replay comparison, private checkpointing, ordered
  publication, and crash-idempotent recovery. These are strong mechanism and reliability facts.
- The stage thresholds deliberately differ. The first transition is allowed while the small network still gains
  appreciably because the medium candidate needs time to catch up; the final transition waits for a much flatter
  medium-model curve.
- The 1.5 multiplier addressed a concrete failure of one-quantum catch-up, but its first implementation indexed the
  fractional schedule by the candidate's own advancing clock. It converged to two quanta per generation rather than
  alternating one and two. Indexing by the outer generation now makes the long-run average exactly 1.5.
- The original catch-up schedule decayed to 0.01, exactly the floor already reached by the active model. A later
  campaign-specific configuration raised the candidate floor to 0.03 so it retained an optimization-rate advantage.
- Loss-based promotion was rejected after it promoted a from-scratch larger candidate that subsequently lost about
  270 Elo. The candidate received more presentations of each replay sample because of its step multiplier, so lower
  training loss was not comparable evidence of equal playing strength. The deployed engine passed its fidelity
  checks, ruling out an INT8 conversion failure as the explanation. Promotion is now match-based.
- Function-preserving growth solved a different problem: instead of asking a random larger model to relearn the
  parent's function, it began at the parent's behavior and exposed only the added capacity to learning.
- Retention is an assembled system decision. There is no fixed-model equal-cost counterfactual proving the causal
  Elo-per-currency gain of this exact sequence, start controller, or promotion controller.

### Pitfalls and interpretation

- “Six-observation window” can be misread as six stored points. The constant is six slope intervals and the runtime
  retains seven EMA samples to compute the oldest-to-current change.
- A slope equal to the threshold does not confirm a plateau; the comparison is strictly below.
- The failed loss gate compared the assembled weighted objective, not playing strength, after unequal optimization
  exposure. It must be presented as a failed controller, not as an ablation showing that large models are weaker.
- Candidate extra quanta increase wall-clock time and repeat the captured boundary's deterministic samples; they do
  not create fresh self-play data or extra replay credit.
- The function-preserving growth tools were an explicit recovery experiment, not yet the initialization path inside
  the general progressive controller.
- The standalone final YAML names the match-gate evaluation but currently omits the corresponding
  `progressive_candidate` evaluation definition. The campaign continuation configuration contains it, while the
  typed loader does not validate the cross-reference. This is a reproducibility defect to fix before reusing the
  standalone recipe, not evidence against the gate itself.

### Unresolved evidence

- No fixed 12-by-128, fixed 14-by-160, or fixed 19-by-176 run provides an equal-cost counterfactual to the full policy.
- The independent contribution of the 50/4 thresholds, six-interval smoothing, two confirmations, 1.5 catch-up
  multiplier, catch-up learning-rate floor, and match threshold has not been ablated.
- The larger grown network reached parity and then remained flat. The reported checkpoint is therefore the retained
  14-by-160 model, not the promoted 19-by-176 continuation. This bounds the result to the tested recipe: added
  capacity was not the binding constraint under the same learning-rate floor, self-play targets, and replay stream.
  It is not a general upper bound on larger networks.

### Sources

- [Authoritative final configuration](../../../py/configs/production/chess-final-config.yaml)
- [Current progressive controller](../../../py/src/training/progressive.py)
- [Training-session integration](../../../py/src/training/session.py)
- [Progressive controller tests](../../../py/test/test_progressive_model_sizing.py)
- [Progressive throughput benchmark](../../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md)
- [Current architecture guide](../../architecture/progressive-model-sizing.md)
- [Function-preserving growth tool](../../../py/tools/grow_checkpoint.py)
- [Grown-checkpoint trainer](../../../py/tools/train_grown_checkpoint.py)
- [QAT reconstruction for grown checkpoints](../../../py/tools/quantize_grown_checkpoint.py)

## Attention and proposed hybrid trunks

### Question

- Does direct square-to-square interaction justify replacing the convolutional residual trunk?

### Approaches and mechanisms

- The implemented attention trunk converts the 8-by-8 board into 64 tokens, applies an input projection plus learned
  row and column embeddings, then uses pre-normalized self-attention and GELU feed-forward blocks.
- The original generic multi-head module was replaced by one packed query/key/value projection and direct scaled
  dot-product attention.
- Three attention-bias choices exist in code:
  - none;
  - a learned relative row/column-offset table;
  - Smolgen-style input-dependent biases generated through a shared template bank.
- Controlled cells tested no bias and Smolgen with dense or from-to heads. The relative-bias implementation does not
  have a documented completed efficacy comparison.
- A CNN/attention hybrid was proposed, but no separate hybrid trunk or alternating-block production experiment was
  found. It must be labeled proposed, not rejected.

### Evidence and results

- Packed query/key/value projection improved short attention training throughput by 14.6%, reducing but not removing
  the deficit to the CNN. Memory and batch-64 inference did not improve in that test.
- The controlled study found:
  - dense-head attention 0.1318 nats worse than the dense-head CNN;
  - from-to attention 0.0060 nats worse than the from-to CNN;
  - Smolgen improving from-to attention by 0.0151 nats;
  - the best Smolgen attention cell 0.0090 nats ahead of the parameter-matched from-to CNN, while delivering only
    36.1% of the dense CNN reference's batch-512 forward rate and 45.8% at batch 64, and using 5.17 times the dense
    CNN reference's peak training memory.
- Early online attention failures are invalid architecture comparisons because generation-zero policies were nearly
  uniform in one path and near-one-hot in another, while ingestion and runtime differed.

### Decision rationale

- Pure attention was implemented and investigated, then not promoted. The modest final proxy advantage of the best
  Smolgen cell did not compensate for its large serving and memory cost under the project's self-play-bound workload.
- Packed query/key/value and bias work are superseded implementation findings because the attention trunk itself was
  not retained.
- Hybrid trunks remain untested proposals.

### Pitfalls and unresolved evidence

- Fused attention operations were invisible to the initial FLOP counter, making attention appear compute-neutral.
  Forcing decomposed attention exposed the missing `64 x 64` score work.
- Backend rankings changed between hardware, batch size, eager/compiled/TorchScript paths, and attention shapes.
- The best-cell comparison has one seed and a shortened horizon. There is no equal-compute online match between the
  best CNN and best attention cell.

### Sources

- [Current attention implementation and bias variants](../../../py/src/training/network.py)
- [Packed-query/key/value benchmark](../../benchmarks/chess-attention-packed-qkv-rtx3060-20260818/README.md)
- [SDPA backend controls](../../benchmarks/chess-attention-sdpa-backends-rtx4070s-20260818/README.md)
- [Controlled attention viability study](../../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)
- [Historical research backlog](../../history/historical-research-backlog-20260822.md)

## Global context in convolutional trunks

### Question

- How can a local convolutional tower condition move features on whole-board state?

### Approaches and mechanisms

- **No context module:** ordinary residual blocks retain only the receptive field accumulated through convolutions.
- **Squeeze-excitation:** global average pooling produces channel gates through a bottleneck MLP implemented as
  `1x1` convolutions and a sigmoid. This was present in the older model family and remains selectable.
- **Global-pooling residual context:** after the first convolution, one quarter of channels become global features.
  Their board-wide means and maxima are concatenated, linearly projected, and added as biases to the local channels
  before the second convolution.
- Placement is configurable for every block or every second block. The retained CNN uses global pooling every second
  residual block.

### Evidence and decision rationale

- Global pooling was motivated by KataGo's strong external ablation and was incorporated into later successful
  screens. The repository has no clean chess one-variable Elo or held-out-loss comparison against disabled context or
  squeeze-excitation.
- The owner remembers a direct comparison in which global pooling trained faster but did not finish at a clearly
  different performance level. Because the result artifact has not been located, this is qualitative design history,
  not a numerical convergence-rate or strength claim.
- Retention is therefore a motivated bundle decision, not a project-measured standalone strength claim.
- Squeeze-excitation should be described as implemented and historically used, not as a rejected chess alternative;
  no adequate direct chess comparison was found.

### Pitfalls and unresolved evidence

- Global pooling creates floating-point islands in the TensorRT INT8 graph. Reducing its frequency might improve
  quantized throughput, but would change the information path and has not been validated for strength.
- External KataGo effect sizes must not be transferred numerically to this chess system.

### Sources

- [Current context implementations](../../../py/src/training/network.py)
- [External-recipe analysis and transfer limits](../../analysis/reference-recipes-for-a-compute-poor-run.md)
- [INT8 graph inspection](../../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md)
- [Historical context backlog](../../history/historical-research-backlog-20260822.md)

## Policy/value trunk sharing

### Question

- How was representation shared between policy and value, and was separation ever investigated?

### Approaches and mechanisms

- Every located production and controlled network computes one shared trunk feature tensor, then applies independent
  policy, WDL, and auxiliary heads.
- No split-trunk design belonged to the experimental program. Any backlog mention was unpursued speculation, not a
  planned or completed investigation.
- No implemented split-trunk module, configuration, benchmark artifact, or controlled result was found in the current
  tree or the Git history paths searched for this dossier.
- The owner confirms that the project always shared the trunk and never seriously considered a separate policy/value
  trunk. The earlier “trunk-sharing experiment” recollection was a misclassification of policy-head capacity work.
  At one point, a large dense primary policy head plus a second policy-shaped auxiliary consumed much of a roughly
  half-million-parameter model; that was an oversized-head architecture mistake, not trunk separation.

### Decision rationale

- Full sharing is an invariant of the implemented research program, not the winner of an ablation. Publication prose
  should describe the shared trunk directly and should not invent a split-trunk investigation.
- Attention models in the recovered controlled study also shared their entire trunk. Their policy-head comparisons do
  not constitute a sharing-versus-splitting treatment.

### Pitfalls and unresolved evidence

- A shared trunk makes head losses interact. Per-loss gradient-norm instrumentation measures how objectives pull on
  the common representation, but it is not a split-trunk ablation.
- Separate value capacity remains a reasonable untested hypothesis if value learning becomes limiting.

### Sources

- [Shared-trunk forward path](../../../py/src/training/network.py)
- [Shared-trunk gradient instrumentation](../../../py/src/training/trainer/rank.py)
- [Historical candidate list](../../history/historical-research-backlog-20260822.md)

## Value-head representations

### Question

- How should outcome information be represented, and how much head capacity is useful?

### Approaches and mechanisms

- The earliest tagged network used a scalar output with `tanh`.
- The retained representation predicts three logits for win, draw, and loss. Search converts the softmax to expected
  value when it needs a scalar, retaining draw information for training and diagnostics.
- The head architecture is
  `1x1 convolution -> BN -> ReLU -> flatten -> hidden linear -> ReLU -> three logits`.
- The retained spatial reduction uses two channels and a 48-unit hidden layer.
- A 32-channel alternative was tested while keeping the remaining architecture fixed.

### Evidence and results

- The short matched 2-versus-32-channel probe found that the wider head added 97,020 parameters, reduced measured
  training throughput by 1.31%, improved policy loss by 0.00338, worsened WDL loss by 0.00029, and improved total
  loss by only 0.00309 in one seed.
- During attempted recovery of a catastrophically quantized trunk, widening only the value head improved neither the
  binding policy failure nor value fidelity enough. The value head remains floating point in the retained quantized
  design.

### Decision rationale

- The two-channel/48-hidden WDL head was retained because the wider head did not earn its extra capacity and did not
  fix the trunk quantization problem.
- The scalar-to-WDL transition is an architectural lineage fact, but no isolated preserved experiment was found that
  quantifies its strength effect. Do not invent one.

### Pitfalls and unresolved evidence

- The width probe was short, single-seed, and measured total training loss rather than online Elo.
- Value-target design—terminal outcome, searched-value blending, cut adjudication—is separate from value-head
  architecture and belongs in the data/target dossier.

### Sources

- [Current WDL head](../../../py/src/training/network.py)
- [Wider-head matched probe](../../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md)
- [Quantization salvage value-head check](../../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md)
- [Earliest tagged scalar-head implementation](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/blob/v1.0/py/src/Network.py)

## Distillation and compact deployment models

### Question

- Can a much smaller network imitate a strong teacher closely enough that additional searches recover the lost
  policy quality at equal serving cost?

### Teacher-output imitation program

- Positions were generated by policy-only teacher self-play, with random opening plies and random legal-move
  perturbations for diversity. The label stored the teacher's legal-masked policy, truncated to its top 64 entries
  and renormalized, plus the teacher's WDL distribution. No tree search generated these labels.
- Students trained with soft policy cross-entropy and WDL cross-entropy. No auxiliary head was distilled. Every
  student used a compact rank-16 dense policy head so head size stayed roughly flat while depth and width varied.
- Dataset and schedule dominated capacity:
  - one million positions became data-bound; longer training increased train/held-out separation;
  - six million positions allowed longer training and reduced the gap substantially;
  - increasing the student from roughly 0.48 million to 1.33 million parameters bought essentially nothing on the
    small dataset but helped once more data was available.
- The strongest 1.33-million-parameter student trailed its roughly 6.06-million-parameter teacher by 176.1 Elo at
  25 searches each and 38.4 Elo when the student received the measured shallow equal-compute budget of 58 searches
  against 25. At 250 searches each, the gap widened to 257.6 Elo.
- The same pattern held for the smaller student: giving both sides ten times more search widened its equal-search
  deficit from 246.3 to 470.4 Elo. Better teacher priors compound under deeper search; search did not simply wash out
  the student's approximation error.

### Replay-target compression program

- The second program did not imitate raw teacher outputs. It trained compact networks on a frozen ten-million-row
  replay slice containing sparse MCTS visits, legal actions, outcome WDL, root value, sample weight, policy surprise,
  and provenance. It is therefore search-target/replay compression, not teacher-logit distillation.
- Four approximately half-million-parameter CNN shapes were trained with two seeds for 100,000 optimizer steps. The
  selected 8-by-56 student had 474,069 deployed parameters, 13.20 times fewer than the 6,256,365-parameter teacher.
- Match outcomes separated three resource questions:
  - at 64 searches each, the student was 291.3 Elo behind;
  - under a saturated measured equal-time workload, the student received 186 searches against 64 and was 166.2 Elo
    behind;
  - at equal network multiply-accumulate count, the student received 850 searches and scored statistical parity,
    but that arithmetic comparison ignores CPU tree work, launch costs, and imperfect batching and is not equal time.
- The student was published as a usable compact artifact. It is a successful compression result, but it did not
  replace direct training or become part of the final self-play recipe.

### Terminal replay-compression saturation check

- A later 6-by-64 student with 470,295 parameters was trained on the terminal 20-million-row replay buffer without
  QAT and evaluated through TorchScript. It is 13.4 times smaller than the 6,315,378-parameter reported teacher.
- One training ended at 36,621 optimizer steps, roughly 7.5 epochs; a second continued to 110,000 steps, roughly 23
  epochs, with the same batch size and learning rate. Their authoritative identities are the inference-model hashes
  in the result files because both were staged through the same evaluation directory.
- At 10,000 searches against the same 20,000-node opponent, the shorter training scored 0.475 (31/33/36), or 2,683
  conditional benchmark Elo, and the longer training scored 0.495 (35/29/36), or 2,697 Elo. The central difference
  is 14 Elo, far inside the paired-match intervals; tripling optimization did not produce a measurable playing gain.
- The training proxy agrees with saturation under this fixed capacity and replay buffer. The policy gap above floor
  improved from 0.5413 to 0.5183, but the long run's held-out policy loss settled near 1.8815 versus training loss
  1.8613 and was nearly flat after about 60,000 steps. This is a bounded saturation result, not proof that all small
  students saturate after one-digit epochs or that more/diverse data could not help.
- The longer student reached 2,873 conditional Elo at 100,000 searches against the 20,000-node opponent, but the
  0.730 score is unbracketed because the planned stronger-opponent match was skipped. It is a lower-bound-style
  point, not a calibrated deep-search headline.

### Decision rationale

- Both programs establish that compact students can recover substantial teacher behavior and that more data matters
  more than a small architectural sweep when the student is data-bound.
- The terminal saturation check adds a stopping lesson: once held-out loss and playing strength stop responding,
  simply tripling passes over one fixed buffer is not supported as an effective compression strategy.
- They also reject the simple deployment thesis that a much smaller model plus proportionally more search will
  necessarily match the teacher. The realizable search multiplier was far below the parameter or MAC ratio, and the
  deficit tended to grow with search depth.
- Distillation remains a useful publishing/compression path, not a retained stage of the primary training algorithm.
  Any future attempt should distinguish raw-output imitation from replay-target compression because their labels and
  questions differ.

### Pitfalls and unresolved evidence

- The teacher-output dataset sampled at a fixed even ply stride, so every retained position had the same side-to-move
  parity. Canonical inputs bounded but did not eliminate the distribution defect. Random independent ply sampling is
  required for a repeat.
- The teacher-output program used a weaker teacher with a known endgame-conversion problem. Its absolute Elo gaps do
  not transfer directly to the final model.
- Equal-compute ratios were noisy under live contention. A ratio must be pinned from an idle, saturated measurement
  at the same root population and batch regime as the match.
- The replay-compression student saw only the final retained ten-million-row window, while the teacher learned from
  the much larger stream that had already been evicted. The result is not a capacity upper bound for that student.
- Only the selected replay-compression seed received matches. Match intervals omit training-seed variance.
- The two earlier programs did not measure the compact student's absolute Stockfish Elo at the intended deep-search
  budgets. The later terminal student did, but its deepest point is unbracketed. The earlier equal-MAC parity result
  must not be described as practical equal-time parity.

### Sources

- [Teacher-output imitation benchmark](../../benchmarks/chess-distillation-probe-rtx3060-20260827/README.md)
- [Replay-target compression benchmark](../../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md)
- [Published compact student artifacts](../../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/model/)
- [Distillation implementation tests](../../../py/test/test_distillation.py)
- Local checksum-verified terminal student evidence:
  `C:\Projects\AZ\.codex-diagnostics\final-2026-09-23\evidence-tail.tgz`, especially
  `distilled-student-36k.log`, `distilled-student.log`, `final-evaluation/student2-vs-20k/result.json`, and
  `final-evaluation/long-student-s10k-vs-20k/result.json`
- Final retrospective supplied on 2026-09-23: `C:\Users\berti\Downloads\RECAP.md`

## Quantization-driven residual architecture

### Question

- Can the trunk constrain activation growth enough for INT8 QAT while still compiling into efficient TensorRT
  residual tactics?

### Approaches and mechanisms

- **Ordinary post-activation residual block:** TensorRT fuses it efficiently, but the trained float network developed
  activation ranges rising from about 0.78 near the input to about 40-43 late in the tower. Full-trunk PTQ was fast
  and behaviorally unusable.
- **Scaled pre-activation block:** `BN -> capped activation -> convolution` branches, residual branch multiplied by
  `1/sqrt(depth)`, and no mandatory post-add activation. It bounded the learning graph but inserted normalization,
  clipping, scaling, additions, and requantization between TensorRT convolution tactics.
- **Scaled post-activation block:** preserves
  `convolution -> BN -> capped activation -> convolution -> BN -> scaled add -> capped activation`. Branch scaling
  can be absorbed into the second convolution at export, retaining more of the conventional fused layout.
- Activation cap 6 and inverse-square-root depth scaling are retained. Initialization was also changed so scaled
  residual towers start with controlled residual contribution.

### Evidence and results

- Full-trunk PTQ reached about three times the TorchScript model-core rate but only 13.3-25.9% policy top-one
  agreement depending on calibration. Weight-only INT8 was accurate but slower than TensorRT FP16.
- The pre-activation graph matched TorchScript throughput before export but expanded from 83 to 330 TensorRT FP16
  layers and to 470 INT8 layers, with 70/102 reformats. Its INT8 core rate was 1.874 times TorchScript rather than
  2.995 times for the original graph, and it still failed fidelity.
- Scaled post-activation QAT learned the replay target normally. A deployment-form continuation recovered useful
  fidelity, and the production-sized smoke reached roughly 135k INT8 positions/s versus 60k TorchScript BF16 and 99k
  TensorRT FP16. The projected end-to-end self-play gain was about 1.7 times before the native TensorRT path existed.
- A numerically shared residual scale was faithful but slower than TensorRT FP16. Removing the residual multiply and
  final clip restored some fusions but did not keep second-convolution outputs in INT8, so throughput barely changed.

### Decision rationale

- Scaled pre-activation was implemented and rejected for this deployment target: both fidelity and practical
  throughput failed.
- Scaled post-activation was retained because it could learn under QAT and preserve a materially faster deployable
  graph after folding and continuation.
- The retained blocks are an architecture/deployment decision. There is no evidence they are intrinsically stronger
  than ordinary residual blocks in floating-point chess.

### Pitfalls and unresolved evidence

- Folding batch normalization only after QAT damaged agreement; training had to continue in the folded deployment
  topology to recover.
- TensorRT graph layer counts and isolated core throughput do not equal end-to-end self-play throughput.
- Global-pooling blocks remain floating-point islands, and policy/value/linear heads are excluded from INT8.
- The final large-model engine identity and precision provenance still require archival verification.

### Sources

- [Quantization salvage investigation](../../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md)
- [Pre-activation architecture screen](../../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md)
- [Scaled post-activation replay screen](../../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md)
- [Current residual-block implementations](../../../py/src/training/network.py)

## Architecture initialization and bootstrap policy shape

### Question

- Why is network initialization part of the self-play algorithm rather than an internal implementation detail?

### Mechanisms and evidence

- The initial network creates the first search priors and therefore the first replay targets. A nearly uniform policy
  does not break search symmetry; a near-one-hot random policy overcommits search to arbitrary moves.
- The failed attention comparison measured these two extremes simultaneously: attention top-three mass was about
  0.11 while the CNN was effectively 1.0 on the probe subset.
- The current path uses architecture-appropriate initialization, a small final policy projection, a final LayerNorm
  for attention, deterministic construction, and calibration on real encoded probes toward a target policy shape.
- Calibration can sharpen or dampen the output. It is not a fixed multiplier associated with one architecture.
- A separate audit found that the configured random seed did not originally reach model construction, invalidating
  supposedly matched independent arms. That defect was repaired.

### Decision rationale

- Bootstrap calibration and deterministic construction are retained correctness controls. They prevent architecture
  comparisons from being dominated by accidental generation-zero policy entropy.

### Pitfalls and unresolved evidence

- Calibrating policy scale does not make initial policies knowledgeable; it only controls concentration.
- Eval-mode BatchNorm on an untrained model can itself distort a probe. Measurements must state whether they use the
  exported inference artifact, training mode, or populated running statistics.

### Sources

- [Controlled bootstrap measurements](../../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)
- [Initialization failure diagnosis](../../plan/chess-post-four-day-regression-analysis-20260820.md)
- [Determinism regression audit](../../analysis/v35-v42-regression-audit-20260913.md)
- [Current calibration code](../../../py/src/training/network.py)

## Publication-level conclusions supported by this dossier

- The project tried more than “dense versus from-to.” It implemented dense reduced-action heads, multiple reduced and
  low-rank dense variants, gathered structured plane logits, a direct 4,864-plane action representation, and a
  structured 1,880-action from-to head.
- The from-to head is retained on the strength of the cleanest head-specific evidence. The dense family is
  superseded, while the plane family is underdetermined rather than cleanly rejected.
- Convolutional and pure attention trunks were both implemented. The controlled evidence attributes most of the
  apparent attention gain to the policy head and generated attention bias; the remaining gain did not justify the
  throughput and memory cost. A hybrid trunk was not tested.
- Full policy/value trunk sharing is an implemented invariant, not the winner of a sharing ablation.
- Global pooling is retained; owner recollection says it learned faster without a clear final-strength difference,
  but no artifact supports a quantitative isolated claim.
- The small WDL head beat a wider alternative on cost/benefit in a short proxy, but scalar-versus-WDL lacks isolated
  strength evidence; trunk splitting was not part of the experimental program.
- The current 52-plane representation is precisely specified and symmetry-tested, but its expansion over the older
  input was not strength-ablated.
- Progressive sizing is a durable, recoverable controller with current start and promotion semantics verified
  against code and configuration; its exact strength-per-cost advantage lacks a fixed-model counterfactual.
- Two distinct compression programs were completed. Raw-output imitation and replay-target compression both produced
  useful small students, but neither justified replacing the teacher in the final deep-search workload.
- Quantization was not a backend-only exercise. It caused a residual-block redesign, exposed graph-fusion constraints,
  and required training in the deployment topology.

## Gaps that need either user memory or recovered evidence

- The raw JSON and complete result table from the seven-way dense policy-head bake-off.
- Any implemented CNN/attention hybrid beyond the pure attention family.
- The artifact identity behind the owner's recollected online plane-head underperformance, including whether “96
  planes” refers to a distinct historical layout or a misremembered 76-plane implementation.
- The artifact behind the owner's recollected global-pooling comparison.
- A controlled scalar-value-versus-WDL experiment.
- A long online ablation for the from-to head, next-policy auxiliary, global pooling, or the exact progressive ladder.
- Final archived identity and precision of the large TensorRT deployment artifact.
