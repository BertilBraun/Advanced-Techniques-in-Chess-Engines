# v36 progressive INT8 production design

This is the authored next-run configuration design. It does not authorize a launch, node mutation, template build,
or GPU spend. The canonical configuration is
`py/configs/production/vast-chess-8gpu-progressive-v36-int8.yaml`.
On this branch it resolves to SHA-256
`75a5b9529e80e32d13c7b3027f4c7c567bade4e9e4e614a8aaad1df89bdf5d80`.

## Training policy

| Setting | Resolved value |
| --- | --- |
| Models | scaled-post 12x128, then scaled-post 14x160 |
| Optimizer | Nesterov accelerated SGD, momentum 0.9, weight decay 0.0001 |
| Gradient clipping | 1.0 |
| Active-model learning rate | linear 0.1 at global generation 0 to 0.01 at global generation 1200 |
| Candidate catch-up learning rate | 0.1 |
| Initial and candidate warmup | 5,000 model-local optimizer steps |
| QAT fold | 5,000 model-local optimizer steps |
| Post-fold warmup | a new 5,000-step phase-local warmup after optimizer recreation |
| Batch and quantum | global batch 2,048; 500 optimizer steps per quantum |
| Replay ratio | 6.25 presentations per newly materialized position |
| Replay capacity | 450k, 900k, 1.5M, 2.1M, 3M, 4.5M, 6M, 9M, 12M, and 15M at generations 0, 5, 10, 20, 40, 70, 100, 400, 700, and 1000 |
| Policy/value weights | 1.0 / 1.0 |

The ordinary trainer warmup is model-local, so a candidate initialized after the Elo trigger receives its own full
5,000-step ramp. The QAT fold recreates both DDP and the optimizer. `deployment_warmup_optimizer_steps` therefore
starts a separate ramp at zero after the fold; the first ramp cannot consume it. With 500 steps per quantum, the
initial fold occurs after ten quanta and the deployment ramp occupies the next ten.

The deployment phase inherits the learning rate selected by the training session. This keeps the active model on
the global-generation schedule and the not-yet-active candidate on its explicit 0.1 catch-up rate. QAT does not
replace either with a model-local deployment schedule.

## Candidate start and promotion

The primary 64-search Stockfish ladder drives the existing Elo plateau policy. Its EMA decay is the code-fixed 0.95.
Candidate training latches after two consecutive observations below 15 benchmark Elo per hour. Each observation is
a completed 30-minute evaluation boundary.

Promotion uses paired total-loss EMAs with decay 0.8 and requires ten shared quanta. The candidate may promote when
its EMA is no more than 1.002 times the active model's EMA. A 1.001 ratio is tighter than the credible resolution of
stochastic paired replay loss and could strand a candidate whose practical loss is matched. A 0.2% allowance is
still materially tighter than v34's 0.5% allowance and does not permit a visibly worse candidate.

Once promoted, the 14x160 model uses the learning rate at the run's current global generation. Promotion remains a
one-way transition.

## Preserved v35/v34 policy

The configuration inherits the current self-play search schedule: 300 visits initially, 400 at generation 10, 500
at 50, 600 at 90, and 800 at 1000. Search parallelism remains 2 at 300-400 visits and 4 at 500-800 visits. Openings
remain 50% uniformly sampled from zero through eight random legal plies and 50% regret-prioritized restart states.
Policy-surprise replay sampling, root-value blending, resignation, temperatures, auxiliary targets, and the
30-minute evaluation cadence are unchanged.

Evaluation retains the policy-only and 64-search adaptive Stockfish ladders with 50 opening pairs per point. Its
first eligible generation moves to 10 because the deployment-form batch-64 TensorRT templates become valid only
after the later fold. This is a startup alignment change, not a change to the evaluated ladder.

## Required artifacts before launch

Self-play needs pre-fold and deployment TensorRT templates for each model at production batch size. Evaluation needs
one deployment-form batch-64 template for each model. Their configured paths are explicit under `/workspace/tensorrt`.
All six templates must be built and validated against the exact source revision before approval. The hardware offer,
dependency lock, configuration hash, and source revision must then be bound in a fresh approval file.
