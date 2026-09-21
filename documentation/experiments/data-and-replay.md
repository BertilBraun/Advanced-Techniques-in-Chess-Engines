# Data generation and replay experiments

## Ledger

| Technique | Status | What the evidence establishes | Principal evidence |
| --- | --- | --- | --- |
| Replay reuse / replay ratio | **Retained** | The final recipe uses reuse 4. Earlier ratios changed generation cadence and schedule pace; the repository does not contain a clean final-lineage strength optimum for 4. | [Final config](../../py/configs/production/chess-final-config.yaml), [v39 decomposition](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md), [v34 dynamics](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md) |
| Staged replay growth to 20M rows | **Retained** | The final configuration grows capacity with training. The late-v34 audit motivates avoiding an overly concentrated recent window, but the exact schedule was not isolated against a fixed-capacity control. | [Final config](../../py/configs/production/chess-final-config.yaml), [v34 dynamics](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md), [reference-recipe analysis](../analysis/reference-recipes-for-a-compute-poor-run.md) |
| Policy-surprise replay sampling | **Retained** | The sampler mixes 30% uniform draws with capped surprise weighting. It is part of the final bundle; no isolated online Elo result establishes its causal gain. | [Final config](../../py/configs/production/chess-final-config.yaml), [reference-recipe analysis](../analysis/reference-recipes-for-a-compute-poor-run.md) |
| Sample weights | **Retained** | Typed sample weights are supported and consumed by the objective. The final recipe's principal rows use weight 1.0; surprise affects sampling, not a claim that arbitrary weighting schemes were validated. | [Final config](../../py/configs/production/chess-final-config.yaml), [v8 data audit](../analysis/v8-training-data-comparison-20260826.md) |
| Random opening prefixes | **Retained** | Half of final games begin from uniformly sampled 0–8-ply random openings. This provides opening diversity but lacks an isolated final-lineage ablation. | [Final config](../../py/configs/production/chess-final-config.yaml) |
| Archive restart states | **Retained** | Half of final games start from archived positions selected using branchability, value, age, and remaining-length filters. The machinery is production, but its independent Elo contribution was not isolated. | [Final config](../../py/configs/production/chess-final-config.yaml), [reference-recipe analysis](../analysis/reference-recipes-for-a-compute-poor-run.md) |
| Regret/difficulty prioritization for restarts | **Retained** | Restart selection includes a uniform component and otherwise prioritizes archived positions by value disagreement; candidate actions come from visit mass. This is the project's implemented form of difficult-state prioritization, not a trained RGSC regret model. | [Final config](../../py/configs/production/chess-final-config.yaml), [reference-recipe analysis](../analysis/reference-recipes-for-a-compute-poor-run.md) |
| Calibrated resignation with continuation games | **Retained** | Production begins calibration at generation 70, bounds false non-loss risk, and preserves 20% continuation games. Early evidence verified the audit path; final-run savings and false-resign rate remain to be reported. | [Final config](../../py/configs/production/chess-final-config.yaml), [canary](../benchmarks/resignation-audit-canary-20260723/README.md), [v8 data audit](../analysis/v8-training-data-comparison-20260826.md) |
| Cut-game root-value target | **Retained** | The final recipe censors remaining-length targets on cut games and uses the cut root value. The dedicated benchmark compared target choices and documented the hazards of fast-search cut values. | [Final config](../../py/configs/production/chess-final-config.yaml), [cut-game benchmark](../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md) |
| Parallel replay materialization | **Retained** | Multi-process materialization and a bounded schema-4 store are production infrastructure. Benchmarks establish loader/materializer throughput and operational behavior, not model-strength gain. | [replay pipeline](../architecture/replay-pipeline-rework.md), [materialization rework](../architecture/replay-materialization-rework.md), [replay loader](../benchmarks/replay-loader-20260724/README.md) |
| Model publication every 100 optimizer steps | **Implemented and rejected** | Excess publication overhead outweighed freshness benefit in the historical experiment. | [research ledger](../history/historical-research-backlog-20260822.md) |
| Reanalysis | **Superseded** | A bounded synchronous reanalysis path existed in the older v10 design, but it is not part of the current replay pipeline or final recipe. There is no current controlled efficacy result. | [v10 implementation record](../history/v10-training-quality-implementation.md), [research ledger](../history/historical-research-backlog-20260822.md) |
| Fully asynchronous self-play/training/publication | **Proposed only** | Current training overlaps a selected subset of actors with optimizer work, but retains synchronized publication and credit quanta. Fully asynchronous learning was not implemented or tested. | [research ledger](../history/historical-research-backlog-20260822.md), [final topology](../../py/configs/production/chess-final-config.yaml) |
| Actor/trainer overlap by pausing half the actors | **Retained** | The final topology pauses two of four actors per GPU during training. Pause sweeps showed that the best trade changes with visit regime; this is an operational choice, not universal proof that 50% is optimal. | [Final config](../../py/configs/production/chess-final-config.yaml), [pause trade-off](../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md) |
| TD-error prioritized replay, recency weighting, deduplication | **Proposed only** | These appear as candidates; no completed production experiment supports them. | [research ledger](../history/historical-research-backlog-20260822.md), [reference-recipe analysis](../analysis/reference-recipes-for-a-compute-poor-run.md) |

## Replay reuse is a coupled scheduling parameter

“Replay ratio” is presentations divided by admitted fresh positions. It controls more than statistical reuse because
generations are credit-funded: changing reuse changes how quickly generation-indexed learning rates, visit budgets,
replay capacities, and evaluation boundaries advance. V39's throughput analysis demonstrates the arithmetic: moving
from reuse 6.25 to 8 would require fewer fresh positions per quantum and was projected to increase generations per
hour by 28% at the same actor rate. That is a projection about cadence, not evidence that reuse 8 learns better.

The final run deliberately uses reuse 4. Its terminal archive should report both configured reuse and empirical
presentations/admitted-position reuse so interrupted or rejected materialization cannot be hidden by the setting.

## Growing the replay window

The staged capacity begins at 600,000 rows and grows to 20 million. This avoids allocating the terminal window before
enough distinct data exist, while eventually retaining a wider history. Existing analysis argues that v34's late
training was data/target limited and that a larger window should be funded by actual data production. It does not
provide a matched online comparison of the exact staged schedule. The report should phrase this as a motivated
design retained in the final run, not as a measured multiplier.

## Surprise sampling and “difficult states”

Two mechanisms must not be conflated:

- replay policy-surprise sampling changes which stored rows the trainer sees;
- restart selection changes which archived positions seed new trajectories.

The final sampler reserves 30% uniform probability and caps surprise at 2.0. Restarts separately mix 30% uniform
selection with value-disagreement-prioritized, filtered, branchable candidates. This realizes the project's
practical “prioritize difficult states” direction, but it is not the learned regret network described in the RGSC
literature. The latter remained a research reference, not an experiment in this repository.

## Resignation

The project moved from an aggressive audit-only canary to a calibrated policy. A candidate threshold is selected
from recent triggered games under a 2.5% false-nonloss upper bound, can relax only gradually, and is checked using a
permanent 20% continuation population. Historical data confirm that continuation assignment and telemetry worked,
but the final report still needs the final threshold trajectory, trigger count, continuation outcomes, search saved,
and any false resignations. Until then the correct claim is “deployed with calibration,” not “proved safe at zero
error.”

## Reanalysis and asynchronous self-play

The history contains a bounded synchronous reanalysis implementation, including materialized overrides, but the
current architecture no longer uses it. The later research ledger correctly conditions any return to reanalysis on
measured replay staleness and a comparison against fresh games per unit compute. No such current comparison exists.

Likewise, the runtime overlaps self-play and training: half the actors continue while the trainer owns all GPUs.
That is not fully asynchronous AlphaZero. Credit quanta, checkpoint publication, and actor refresh remain coordinated.
Documentation should use “overlapped self-play and training,” not “asynchronous training.”

## Remaining evidence gaps

- No one-variable online study isolates the final capacity schedule, reuse 4, surprise sampler, restart mixture, or
  random openings.
- Final resignation calibration and savings need extraction from the terminal archive.
- Replay-age percentiles, effective unique-row reuse, rejection rates, and the sampling-weight distribution should
  be plotted for the final report.
- Reanalysis must be described as historical/superseded, and fully asynchronous learning as unattempted.
