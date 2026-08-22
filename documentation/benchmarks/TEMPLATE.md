# <topic> — <hardware>, <YYYY-MM-DD>

<!-- Directory name: <topic>-<hardware>-<YYYYMMDD>/. Hardware is the GPU model and count (8xrtx3060,
     rtx4070s), never a topology string, a provider name, or "local". Date, not an ISO timestamp.
     A benchmark missing any field in the header table below is not evidence.
     Model READMEs: node-comparison-vast-4nodes-20260821/ and naive-python-mcts-rtx3060-20260816/. -->

| | |
| --- | --- |
| `experiment_configuration_sha256` | `<64 hex, from the resolved config>` |
| Source revision | `<full 40-hex SHA>` (`clean` / `dirty: <what>`) |
| Node | `<instance/container id, GPU model+count, driver, effective CPUs, RAM>` |
| Date | `<YYYY-MM-DD>` |

## Method

What was measured, what was excluded, warm-up, sample window, repetitions.

## Results

One table, raw numbers with units. No derived Elo or per-dollar figure without its formula.

## Interpretation

What the numbers do and do not license. Numbers are not comparable across different hardware without
saying so explicitly.

## Reproduce

Exact command line and the tool under `py/tools/` or `deployment/`.

## Files

One line per non-README artifact in this directory.
