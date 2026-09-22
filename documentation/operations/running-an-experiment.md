# Running an experiment

The end-to-end path for one experiment, from writing a configuration to reading its result. `run-control.md`
documents each `run_control.sh` subcommand; this is the order to use them in and the things that go wrong.

Everything up to the push happens on the workstation. Everything after it happens on the node, through
`deployment/remote_command.sh <HOST[:PORT]> <command …>`, which owns the key, user and connection options.

## 1. Write the configuration

Put it in `py/configs/production/` and `extends:` the closest existing arm, so the diff is the experiment.
Lists replace wholesale, so overriding one evaluation definition means restating the block.

## 2. Validate it locally

Resolve the new configuration and the one it is compared against, then diff the two resolved models. This is the
check that catches an unintended change, and it is worth running every time:

```bash
cd py && PYTHONPATH=. python - <<'PY'
import json
from pathlib import Path
from src.experiment.configuration import load_experiment_configuration

base = Path('configs/production')
old = json.loads(load_experiment_configuration(base / '<comparator>.yaml').model_dump_json())
new = json.loads(load_experiment_configuration(base / '<new>.yaml').model_dump_json())

def walk(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            walk(a.get(key, '<missing>'), b.get(key, '<missing>'), f'{path}.{key}')
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f'{path}[{index}]')
    elif a != b:
        print(f'{path}: {a!r} -> {b!r}')

walk(old, new)
PY
```

Read the output and make sure every line is a change you meant.

## 3. Commit and push

```bash
git add py/configs/production/<name>.yaml && git commit && git push
```

Nothing reaches the node except through Git; the node fetches what it runs.

## 4. Prepare the checkout

```bash
bash deployment/run_control.sh prepare <branch> <revision> py/configs/production/<name>.yaml
```

Fetches, checks out, verifies the tree is clean, brings the interpreter and native extension up to the revision,
resolves the configuration hash on the node and writes the approval. Repeating it is safe: the fetch is skipped
when the revision already matches, and a matching approval is left alone.

It refuses while a run from the same checkout is live. Wait for that run to stop, or prepare a different checkout.

## 5. Start

```bash
bash deployment/run_control.sh start py/configs/production/<name>.yaml
```

`start` validates the approval `prepare` wrote, preflights the first TensorRT template at the self-play batch
size, then launches under supervisor.

## 6. Watch it

```bash
bash deployment/run_control.sh status <run-name>
```

Exits non-zero when unhealthy, so a monitor can call it directly. Watch the first evaluation land before trusting
the run: a configuration error surfaces at generation zero, a template or fidelity problem at the first
publication.

## 7. Stop

```bash
bash deployment/run_control.sh stop <run-name>
```

Requests a checkpoint-safe exit and preserves the run. It finishes the current generation first, so it takes
minutes.

## 8. Get the evidence off the node

The node is ephemeral and nothing on it is durable. `preserve` archives configuration, logs, TensorBoard and run
state — **but not weights** — under `.codex-diagnostics/<run-name>-<UTC>/`. Copy that off before the instance goes
away, and verify it against the archive's own `SHA256SUMS`.

## What goes wrong

**Set `allow_fidelity_deviation: true`.** A single probe position exceeding the maximum policy KL gate ends the
run. It ended V84 at generation 21 while top-1 agreement was 0.9875 and mean KL was half its limit. INT8 breaches
these gates routinely and cannot run without it.

**Never delete `optimizer_*.pt`, `model_*.pt` or `*.onnx` from a run you might resume.** A checkpoint manifest
references all three. Reclaiming disk by stripping them turns a resume into a rerun, which is how V84's arm was
lost. `replay.bin` and `completed-games/` are the safe things to reclaim.

**Read the largest log, not the newest.** `ls -t` usually returns `supervisor-stdout.log`, which interleaves every
attempt at the run. The runner's own log is the largest one and is where the failure is.

**A quiet watcher is not a healthy run.** The SSH proxy on rented nodes resets often enough that a `tail -f`
watcher dies silently. After any drop, query the run directly rather than inferring from the absence of alarms.

**Plan the disk before starting.** `replay.bin` is roughly 4.1 KB per sample, so an eight-million-sample buffer is
about 33 GB and the capacity schedule keeps stepping. Check that the whole schedule fits before a long run, not
once it is already filling.

**Let `prepare` write the approval.** It resolves the configuration hash on the node, which is where `start`
checks it. The hash no longer depends on the platform - configuration paths serialise with forward slashes on every
host - but `prepare` also pins the revision and writes the approval in one step, which doing it by hand does not.
