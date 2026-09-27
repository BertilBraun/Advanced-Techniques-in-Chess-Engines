# Appendix D. Reproducibility and release boundary

This appendix specifies the network representation, numerical recipe, and artifacts required to reproduce the
software and evaluation. The released configuration supports new training runs; the published checkpoint fixes
the model used for the reported results.

## Network input and output contract

The network input is a 52×8×8 tensor with channels listed in Table \ref{tab:appendix-d-reproducibility-1}. "Own" denotes the player to move.
For Black to move, ranks are reflected and piece colours are exchanged;
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

Channels 0–39 are binary; channels 40–51 contain unnormalized scalar values. Missing history entries are
zero-filled. Move history stores origin and destination pairs for the last eight plies rather than complete
board states; castling records the king's actual destination. The checkerboard is fixed in canonical coordinates.

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

## Configuration and reported result

The fully expanded `chess-final-config.yaml` is the maintained entry point for training with the current recipe.
Replication of the reported experiment instead uses its frozen source revision, resolved configuration,
manifest, checkpoints, engines, and datasets. This separates future recipe updates from the fixed experimental
record.

## Expanded chess recipe settings

The final recipe ran on one node with eight RTX 4070 SUPER GPUs and 80 logical CPUs. Its network ladder contains
12×128, 14×160, and 19×176 residual CNNs with capped scaled post-activation branches and global-pooling context
in every second block. The reported checkpoint uses the 14×160 stage.

The architecture uses a key-size-128 chess from-to policy head and a
two-channel WDL head with a 48-unit hidden layer. The training-only next-searched-policy and remaining-game-length
heads have loss weights 0.15 and 0.1. Primary policy and value losses each have weight 1.0. Terminal outcome
targets are discounted by 0.998 per ply; the search-root-value blend rises from zero to 0.1 over its configured
schedule, while search backup uses a separate 0.99 per-ply discount.
At a capped game's final position, a searched scalar value `v` is converted to a soft WDL target with
`r = 1 - |v|`: win, draw, and loss receive `max(v, 0) + r/3`, `r/3`, and `max(-v, 0) + r/3`, respectively.
Capped games omit the remaining-game-length loss. Batch-normalization folding, recalibration, ONNX export, and
TensorRT refitting operate on a deployment copy, leaving the trainable model unfused.

Each 500-step training quantum uses eight ranks processing 256 positions each, for a global batch of 2,048.
Nesterov SGD uses momentum 0.9, weight decay 0.0001, and gradient clipping at norm 1.0. The learning rate warms
from zero to 0.1 over the first 1,000 optimizer steps and then follows the configured linear schedule to 0.01.
Training uses bfloat16 and persistent trainer processes; `torch.compile` is disabled. QAT calibration uses 516
real evaluation positions and is refreshed at every publication boundary.

The 32 self-play actors run four per GPU, with 512 interleaved games per actor. Native inference uses batches of
320 with two outstanding batches per worker. Search uses exploration constant 1.5, reduced-parent FPU with
reduction 0.2, forced playout coefficient 1.5, and Dirichlet epsilon 0.25 with alpha 0.3.
Eight materializers convert completed games into the fixed-layout
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

## Self-play schedules and promotion

Half the actors pause during each training quantum. The search schedule rises from 300 to 800 visits per move;
the reported checkpoint lies in the 600-visit stage. Games start approximately equally often from random legal
openings of up to eight plies and from eligible restart states, falling back to a random opening when no restart
is available. Game caps increase from 150 to 250 plies, and greedy move selection starts later as training matures.
Twenty percent of games are designated as no-resignation continuations to calibrate the resignation threshold
against a 2.5% upper-bound target for mistakenly resigning a win or draw.

Each admitted replay position funds four training presentations. Sampling assigns 30% uniform probability and
otherwise prioritizes bounded policy surprise. Candidate training begins when smoothed 64-search ladder gains
fall below 15 Elo/hour for the small stage or 4 Elo/hour for the medium stage. Promotion uses the paired-match
gate specified above. Training monitors run every 20 minutes using 50 paired openings, a three-rung adaptive
Stockfish 13 bracket, and both 64-search and policy-only play.

## Result identity and provenance

The experiment archive records the following provenance:

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

The public source release [10] contains setup and validation instructions. On production nodes,
`deployment/setup_remote.sh` installs locked dependencies, builds the Release extension, installs pinned
evaluation engines, and verifies engine execution. `deployment/run_control.sh` manages the run lifecycle.

Original project code and documentation, including this report, are available under the MIT License in the
source release [10]. The published final model artifacts carry the same license in the model repository [11].
External dependencies, reference sources, and third-party data retain their own terms.

## Reproducing evaluation

Matched evaluation uses the published checkpoint and inference artifact, paired opening suite, opponent binary,
node limits, thread and hash settings, candidate search budget, parallelism, batching, and adjudication rules.
Re-exporting with a different toolchain can change predictions and constitutes a new deployment comparison.
Appendix \ref{app:B} specifies the final match protocol and rating calculation.

Performance measurements distinguish isolated neural-network throughput, saturated multi-game search, and
single-game latency. These measure different execution regimes and require separate benchmarks.

## Reproducing plots and tables

Plots and tables are generated from archived JSON, CSV, TensorBoard, and manifest data, with extraction methods
retained alongside the evidence. The underlying columns support recomputation of totals, rates, Elo estimates,
and uncertainty intervals independently of the live dashboards.
