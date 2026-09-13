# V35-code / V42-initialization controlled A/B

This run isolates the production-code change between the successful V35 lineage and V42. It runs the V35-era
training stack from `1d57b0ab853adef2764e864e87898de1c74f0a37`, but imports the exact V42 generation-zero model, optimizer,
TorchScript inference artifact, QAT state, and bootstrap policy calibration. Replay, completed games, restart states,
credits, evaluations, and TensorBoard state start empty.

`1d57b0ab` is the correct V35 boundary. The original V35 source revision, `8b02c00af25aafc4fe1edb3e16ec2fe5dfb3afa5`,
could train but pruned evaluation inference artifacts before queued matches consumed them. `1d57b0ab` contains only
the evaluation-artifact retention repair needed to obtain valid 20-minute policy-only and 64-search measurements;
its network, QAT runtime, TensorRT publication, training, and self-play behavior are unchanged from `8b02c00a`.

The controlled run uses the fixed 14x160 scaled-post-activation network, SGD/NAG, the V35 QAT lifecycle and learning
rate, replay ratio 6.25, the 15-million-position staged replay capacity, and the V35 search/opening recipe. The two
Stockfish-13 ladder tracks run every 20 minutes. Their first eligible generation remains 2 because V35 cannot publish
the deployment-phase INT8 evaluation engine before its generation-2 QAT fold.

The source checkpoint must first be copied out of the V42 run directory into an immutable seed directory. The copy
tool loads the manifest, verifies every recorded SHA-256, copies all four artifacts atomically through the normal
checkpoint persistence code, rewrites paths relative to the destination, and refuses an existing non-matching seed:

```bash
/workspace/alphazero-engine-venv/bin/python py/tools/copy_checkpoint.py \
  --source-manifest /workspace/alphazero-engine-int8-validation/py/training_data/production/vast-chess-8gpu-fixed-medium-v42-int8/checkpoint_0.json \
  --generation 0 \
  --destination /workspace/controlled-initializations/v42-generation-0
```

The test changes only the source implementation. It deliberately retains V35's warmup rule, which starts at nearly
zero and reaches the scheduled learning rate over 1,000 optimizer steps. V42 added a configurable 0.001 warmup floor;
that behavior is part of the code/configuration interval under test rather than something backported into this arm.
