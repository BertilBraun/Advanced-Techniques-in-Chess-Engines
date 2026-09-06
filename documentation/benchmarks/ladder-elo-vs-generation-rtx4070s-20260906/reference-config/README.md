# v29 resolved configuration, kept for reference

`vast-chess-4day-production-v29-resume.yaml` is the exact configuration the v29 run used, preserved
here because the next run's design needs something to diff against.

It is **a record, not a loadable config**. It carries `search_stopping:` and `search_budget:` blocks
for machinery that was removed from the tree in `71bc0e5c`, so it will not validate against current
code — `FrozenModel` uses `extra='forbid'`. That is why it lives under `documentation/` rather than
`py/configs/`, which `py/test/test_config_tree.py` requires to resolve.

Verified: sha256 `b8944771b1f5039427218a7334acc3d158098c4b12cf1d563c9edd258fe0cf71`, identical to the
`config/vast-chess-4day-production-v29-resume.yaml` entry in the run's own preserved archive manifest
(`vast-chess-4day-production-v29-20260906T124211Z/SHA256SUMS`). So this is the resolved configuration
the run actually executed, not a reconstruction.

## The parameters that matter for the next run

| Parameter | v29 value |
|---|---|
| `global_batch_size` | 2048 |
| `optimizer_steps_per_quantum` | 500 |
| `replay_ratio` | 8 |
| replay `maximum_capacity` | 5,000,000 |
| `baseline_visits` | 1000, cut to 600 at generation 797 |
| progressive model tiers | 12x128 -> 14x160 -> 19x176 |
| final state | generation 1002, 501,000 optimizer steps, 71.3 h |

Measured outcome: 2034 Elo at 2.7 h rising to 2800 at 71.3 h, with 90% of the final strength reached
in the first 28 hours. See the README beside this directory.

The full v29 lineage, including the fork arms and the v21-v30 production configs, is on the
`adaptive-stopping` branch and the `adaptive-stopping-final` tag.
