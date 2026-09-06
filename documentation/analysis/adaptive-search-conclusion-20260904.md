# Adaptive search is finished: conclusion and handoff

2026-09-04. Runs v13-v30 plus the generation-310 fork experiment, all on the 8x RTX 4070 SUPER node
`38.49.42.120`.

Two successive attempts to spend search adaptively are now both closed as negative. This note records
what was measured, what should be kept, and what should be deleted.

## The two attempts

**Predicted per-position budgets** (v15-v20) were retired on 2026-09-01 and are written up in
`adaptive-search-budget-negative-result-20260901.md`. Summary: the implementation met every design
target and still lost 60-100 Elo against runs without it.

**Learned early stopping** (v21-v30) is retired here.

## What the fork experiment measured

Every earlier comparison in this project confounded the intervention with initial weights, replay
contents and early-training dynamics. The fork experiment removed all three: v29's generation-310
checkpoint was copied into a fresh run with an empty replay, ten generations rebuilt a buffer that the
forked model itself produced, and that state was frozen. Three arms then resumed from **byte-identical
copies** of it, differing only in the stopping configuration.

| arm | ceiling | generations | s/gen | credit-wait | spend | final Elo |
| --- | --- | --- | --- | --- | --- | --- |
| c025 | 0.25 | 72 | 144 | 57.2 s | 0.864 | 1827 |
| c015 | 0.15 | 67 | 156 | 65.9 s | 0.990 | 1832 |
| baseline | off | 69 | 149 | 80.5 s | - | 1894 |

Paired at matched generations, nine points each:

    baseline - c025   +1.7 Elo  (SE 9.9, t +0.17)
    baseline - c015   -4.2 Elo  (SE 10.1, t -0.42)

**No detectable difference.** The mechanism worked exactly as designed - credit-wait ordered
57 < 66 < 80 seconds precisely by how much search each arm skipped - and none of it converted to
strength.

## Why the throughput argument fails

The headline saving is misleading. c025 skipped **14% of search** but ran only **3% faster per
generation** (144 s against the baseline's 149 s), because self-play overlaps with training: sixteen of
thirty-two workers run during the trainer quantum, so only the non-overlapped remainder appears as
credit-wait. Cutting self-play cost shrinks the slack, not the critical path.

At the measured 0.26 Elo per generation in this regime, a 3% cadence gain is worth **about 1 Elo over
three hours** - an order of magnitude below the +/-10 Elo resolution of the best-controlled experiment
this project can run. The benefit is not merely unproven; it is too small to be provable.

## What was learned that outlives the negative result

**Regime matters more than we assumed.** Every tuning decision before this had been measured in the
first few hours of a run and extrapolated. That was wrong repeatedly: the self-play pause sweep
inverted twice between the 300-visit and 600-visit regimes, and v30's stopping never left its
calibration transient in three hours, so the "stopping does nothing" reading from it was measuring the
warm-up rather than the technique. From a trained checkpoint the rule found a stable operating point
within one generation.

**Forking from a shared state is the only comparison that works here.** Independent runs have a
per-bucket ladder noise of about 41 Elo; paired arms from an identical state have a paired-difference
noise near 10-24 Elo. Every future A/B should fork rather than start fresh.

**`training.random_seed` never reached the network.** It seeded the self-play workers and the
evaluation dataset but not model construction, so no run in this project was reproducible and the
initial policy entropy ratio varied 0.816-0.898 across configs that were otherwise identical. Fixed in
`d52949c7`.

**The trunk gradient probe never produced a number.** Under `torch.compile` the trunk activation is a
sibling graph output with no autograd path from the loss, so `autograd.grad` returned `None` and
`allow_unused=True` silently substituted zeros - for every term, in every compiled run. Fixed in
`60abc966`, though the probe is disabled in production because the reworked version takes gradients
over DDP-wrapped parameters and that interaction is unvalidated.

## What should merge to master

Independent of adaptive search, and worth keeping:

- `d52949c7` seed the trainer network initialization - restores reproducibility
- `c13b749c` keep loaded AdamW step counters on the CPU - correctness, with a CUDA-gated test
- `35b00796` reply before loading the model on checkpoint refresh
- `eefb5d54` cache the stop-calibration audit window in memory - only useful while stopping exists
- `14cd724f` make `run_control fetch` work - the previous implementation used rsync and had never
  succeeded from the workstation
- `68cf04ba`, `6fa851df` the self-play/trainer throughput benchmark and its worker-spread fix
- `d2ffdaba` the self-play pause trade-off benchmark write-up
- `60abc966` the gradient probe fix, kept disabled

The production configurations should be squashed to a single current template rather than merged as
the v13-v30 sequence.

## What should be deleted

| area | lines |
| --- | --- |
| `py/src/search_stopping/` | 1,579 |
| `py/test/test_search_stopping_*.py` | 725 |
| native stopping paths across 7 files in `cpp/` | up to 2,490 |
| stopping references in 14 other `py/src` modules | - |

Also removable: `search_stopping` configuration blocks in every config, the `policy_checkpoint_visits`
contract between Python and the native search, the audit-record plumbing, and the stop-policy
publication path through `SelfPlayGroup`.

The deep-search labelling infrastructure, TorchScript export/consumption and telemetry were listed as
reusable when predicted budgets were retired. With stopping now retired too, nothing consumes them and
they should go with it.

## Operational cost of the two attempts

Seven run-killing defects across the series: duplicated configuration pins (v15), an inverted isotonic
projection (v17), a dual seeded from measured curves (v19), a corrector standardising against an
absolute floor, a missing checkpoint-visits contract that killed all 32 workers instantly (v21), and a
NaN loss from the reworked gradient probe on a resumed run. Every one was caught only in production.

## Status

Adaptive search - predicted budgets and learned stopping alike - is closed. v29 continues as the
production run with stopping disabled.
