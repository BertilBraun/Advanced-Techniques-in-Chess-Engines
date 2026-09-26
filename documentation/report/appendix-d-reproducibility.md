# Appendix D. Reproducibility and release boundary

## Network input and output contract

The network input has shape 52×8×8. Table D1 specifies every channel; channel ranges list features in tensor
order. "Own" means the player to move. For Black to move, ranks are reflected and piece colours are exchanged;
files retain their order. Tensor row zero is the player's home rank, and column zero is file a. There is no separate
absolute-colour plane. Spatial masks contain zeros and ones; flags and scalar values fill all 64 squares.

| Plane indices | Feature, in channel order | Encoding |
| --- | --- | --- |
| 0, 1, 2, 3, 4, 5 | Own pawn, knight, bishop, rook, queen, king | Piece-location masks |
| 6, 7, 8, 9, 10, 11 | Opponent pawn, knight, bishop, rook, queen, king | Piece-location masks |
| 12 | Own kingside castling right | Constant 0 or 1 |
| 13 | Own queenside castling right | Constant 0 or 1 |
| 14 | Opponent kingside castling right | Constant 0 or 1 |
| 15 | Opponent queenside castling right | Constant 0 or 1 |
| 16 | All own pieces | Occupancy mask |
| 17 | All opponent pieces | Occupancy mask |
| 18 | Pieces checking the player to move | Checker-location mask |
| 19 | En-passant target | One square, or all zeros |
| 20 | At least one earlier occurrence of this position | Constant 0 or 1 |
| 21 | At least two earlier occurrences of this position | Constant 0 or 1 |
| 22, 23 | Most recent move: origin, destination | Two single-square masks |
| 24, 25 | Second-most-recent move: origin, destination | Two single-square masks |
| 26, 27 | Third-most-recent move: origin, destination | Two single-square masks |
| 28, 29 | Fourth-most-recent move: origin, destination | Two single-square masks |
| 30, 31 | Fifth-most-recent move: origin, destination | Two single-square masks |
| 32, 33 | Sixth-most-recent move: origin, destination | Two single-square masks |
| 34, 35 | Seventh-most-recent move: origin, destination | Two single-square masks |
| 36, 37 | Eighth-most-recent move: origin, destination | Two single-square masks |
| 38 | Fixed checkerboard, with a1 set to one | Alternating 0/1 mask |
| 39 | Exactly one bishop per side, on opposite colours | Constant 0 or 1 |
| 40, 41, 42, 43, 44, 45 | Pawn, knight, bishop, rook, queen, king balance | Own count minus opponent count |
| 46 | Halfmove clock | Integer count, capped at 100 |
| 47, 48, 49, 50, 51 | Own pawn, knight, bishop, rook, queen counts | Integer counts |

The first 40 planes are binary; the remaining 12 are scalar planes. Counts are not divided by their maxima.
Missing history entries are zero-filled. History records the last eight moves, not eight full board states;
castling records the king's actual destination. The checkerboard is fixed in canonical tensor coordinates.

The shared backbone produces 160×8×8 features. The policy head projects each square to a 128-dimensional token
and forms query and key vectors whose scaled dot products score origin-destination pairs. These scores are gathered
into 1,880 action logits: 1,792 ray or knight pairs and 88 explicit promotion actions. Promotion offsets distinguish
queen, rook, bishop, and knight choices. Castling uses the king-to-own-rook pair in the action encoding, while
en passant uses the pawn's ordinary origin-destination pair. Illegal actions are masked before softmax over the
legal moves.

The value branch uses a two-channel 1×1 convolution, batch normalization, ReLU, flattening, a 48-unit hidden layer,
and three logits. Softmax produces win, draw, and loss probabilities for the player to move. The auxiliary branches
produce a further 1,880 logits for the next searched policy and one scalar for remaining game length. They are used
only in training; inference returns the primary policy logits and WDL probabilities.

## Two reproducibility targets

Reproducing the current system and reproducing the reported result require different starting points:

1. **Recipe reproduction:** use the current fully expanded `chess-final-config.yaml` as the supported entry point.
2. **Result reproduction:** use the frozen source revision, resolved config hash, manifest, checkpoints, engines,
   datasets, and archive recorded for the final result.

The recipe may evolve; the reported result remains fixed.

## Expanded chess recipe settings

The architectural settings summarized in Chapter 7 include a key-size-128 chess from-to policy head and a
two-channel WDL head with a 48-unit hidden layer. The training-only next-searched-policy and remaining-game-length
heads have loss weights 0.15 and 0.1. Primary policy and value losses each have weight 1.0. Terminal outcome
targets are discounted by 0.998 per ply; the search-root-value blend rises from zero to 0.1 over its configured
schedule, while search backup uses a separate 0.99 per-ply discount.
At a capped game's final position, a searched scalar value `v` is converted to a soft WDL target with
`r = 1 - |v|`: win, draw, and loss receive `max(v, 0) + r/3`, `r/3`, and `max(-v, 0) + r/3`, respectively.

Each 500-step training quantum uses eight ranks processing 256 positions each, for a global batch of 2,048.
Nesterov SGD uses momentum 0.9, weight decay 0.0001, and gradient clipping at norm 1.0. The learning rate warms
from zero to 0.1 over the first 1,000 optimizer steps and then follows the configured linear schedule to 0.01.
Training uses bfloat16 and persistent trainer processes; `torch.compile` is disabled. QAT calibration uses 516
real evaluation positions and is refreshed at every publication boundary.

The 32 self-play actors run four per GPU, with 512 interleaved games per actor. Native inference uses batches of
320 with two outstanding batches per worker. Search uses exploration constant 1.5, reduced-parent FPU with
reduction 0.2, forced playout coefficient 1.5, and Dirichlet epsilon 0.25 with alpha 0.3. Restart-state
selection retains a 30% uniform component. Eight materializers convert completed games into the fixed-layout
memory-mapped replay store; training credit is committed only against durable admitted rows.
Restart positions require at least 15 plies remaining in the source game, absolute root value at most 0.8, and
two or three leading actions covering 85% of visit mass. The played branch is marked used, and reservations keep
workers from claiming the same alternative concurrently. Replay's logical capacity grows through 0.6, 1.2, 2.0,
2.8, 4, 6, 8, 12, 16, and 20 million rows.
Restart selection is 30% uniform and otherwise weighted by the square root of value correction, defined as half
the absolute difference between searched root value and raw network value. Age and capacity bounds remove old
states, and the archive is local to each worker.

The successor network trains on a captured replay snapshot for an average of 1.5 optimizer quanta per
active-model quantum, with its own catch-up learning-rate clock. The configured match gate requires the successor
to score at least 0.48 in two consecutive paired evaluations. The fully expanded config [10] specifies the
stage-specific ladder-plateau thresholds and window.

## Result identity and provenance

The frozen local result record identifies these layers of provenance:

- Git revision and clean/dirty state, resolved YAML and SHA-256, and dependency-lock hash;
- operating image, software and driver versions, GPU, CPU, RAM, disk, and engine versions and archive hashes;
- evaluation dataset and opening-suite identities, hashes, raw matches, aggregates, commands, and interval method;
- run manifest, approval, coordinator logs, TensorBoard events, and resource telemetry;
- replay schema, capacity, occupancy, and reconciled volume counters;
- checkpoint and inference hashes, ONNX and TensorRT provenance, calibration positions, and fidelity reports;
- one digest covering the fetched archive or a checksummed artifact manifest.

The published INT8 ONNX has SHA-256 `d634abacae3c874eac6ded89f6af861eb81b509da638b5ad710587b1a08be658` at
the immutable model-repository revision `dc8fccccb67ab5ec9e36267a165a9700b7dbf55f` [11]. The source release's
evidence index [10] records additional artifact hashes. The frozen training-source revision, resolved configuration,
large run archives, and some exact evaluation inputs remain in the local result archive. The public recipe and model
support inspection and a new run, but are not yet a self-contained package for exact-match reproduction of the
reported experiment.

## Reproducing the software

Local setup and validation begin in the public source-code release [10]. Production nodes are provisioned by
`deployment/setup_remote.sh`, which installs locked dependencies, builds the
Release extension, installs pinned evaluation engines, and runs engine smokes. Run lifecycle operations go through
`deployment/run_control.sh`.

Original project code and documentation, including this report, are available under the MIT License in the
source release [10]. The published final model artifacts carry the same license in the model repository [11].
External dependencies, reference sources, and third-party data retain their own terms.

## Reproducing evaluation

Use the preserved checkpoint and inference artifact rather than re-exporting it with a newer toolchain. Reuse the
same paired opening suite, colors, opponent binary, node limit, threads, hash, candidate search budget, parallelism,
batching, and adjudication rules. Report every game and recompute aggregates independently.

For latency, separate:

- isolated model-forward throughput;
- saturated many-position search throughput;
- single-game interactive latency.

Only compare like with like. Appendix B states the terminal opponent, openings, game count, and inference artifact.

## Reproducing plots and tables

The plots derive from archived JSON, CSV, TensorBoard, or manifest data. Figure inputs and extraction methods are
retained beside the final evidence archive. Derived tables preserve the raw columns needed to recompute totals,
rates, Elo transformations, and uncertainty intervals; live-dashboard values are not treated as publication data.
