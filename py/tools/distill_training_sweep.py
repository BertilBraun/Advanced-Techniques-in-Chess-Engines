from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.util.atomic_file import write_text_atomically

HEADLINE_PATTERN = re.compile(r'headline policy gap above floor ([0-9.]+)')
POLL_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True)
class SweepArm:
    name: str
    layers: int
    hidden_size: int
    extra_arguments: tuple[str, ...]


# Arms 1-2 are the references, 3-4 ask whether the dataset is still the binding constraint, 5-6 test spending the
# schedule at a high rate, and 7-8 sit at one size so that they and arm 1 isolate each auxiliary head in turn.
SWEEP_ARMS = (
    SweepArm('6x64-baseline', 6, 64, ()),
    SweepArm('8x96-baseline', 8, 96, ()),
    SweepArm('6x64-half-data', 6, 64, ('--training-fraction', '0.5')),
    SweepArm('8x96-half-data', 8, 96, ('--training-fraction', '0.5')),
    SweepArm('6x64-plateau', 6, 64, ('--learning-rate-schedule', 'plateau', '--anneal-fraction', '0.2')),
    SweepArm('8x96-plateau', 8, 96, ('--learning-rate-schedule', 'plateau', '--anneal-fraction', '0.2')),
    SweepArm('6x64-aux-both', 6, 64, ('--distil-auxiliary-heads', 'next_policy', 'remaining_game_length')),
    SweepArm('6x64-aux-next-policy', 6, 64, ('--distil-auxiliary-heads', 'next_policy')),
)


@dataclass(frozen=True)
class SweepArguments:
    dataset: Path
    output_root: Path
    generation: int
    steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    evaluate_every: int
    checkpoint_every: int
    policy_bottleneck_rank: int
    device_ids: tuple[int, ...]
    random_seed: int
    dry_run: bool


@dataclass
class RunningArm:
    arm: SweepArm
    device_id: int
    process: subprocess.Popen
    log_path: Path
    started_at: float


def build_command(arm: SweepArm, arguments: SweepArguments, device_id: int) -> list[str]:
    return [
        sys.executable,
        '-m',
        'tools.distill_train_student',
        '--dataset',
        str(arguments.dataset),
        '--output-run-state',
        str(arguments.output_root / arm.name),
        '--layers',
        str(arm.layers),
        '--hidden-size',
        str(arm.hidden_size),
        '--policy-bottleneck-rank',
        str(arguments.policy_bottleneck_rank),
        '--batch-size',
        str(arguments.batch_size),
        '--steps',
        str(arguments.steps),
        '--learning-rate',
        str(arguments.learning_rate),
        '--warmup-steps',
        str(arguments.warmup_steps),
        '--evaluate-every',
        str(arguments.evaluate_every),
        '--checkpoint-every',
        str(arguments.checkpoint_every),
        '--generation',
        str(arguments.generation),
        '--random-seed',
        str(arguments.random_seed),
        '--device-id',
        str(device_id),
        *arm.extra_arguments,
    ]


def completed_arms(arguments: SweepArguments) -> tuple[str, ...]:
    return tuple(
        arm.name
        for arm in SWEEP_ARMS
        if (arguments.output_root / arm.name / f'checkpoint_{arguments.generation}.json').exists()
    )


def final_headline_gap(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    matches = HEADLINE_PATTERN.findall(log_path.read_text(encoding='utf-8', errors='replace'))
    return float(matches[-1]) if matches else None


def run_sweep(arguments: SweepArguments) -> int:
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    already_done = completed_arms(arguments)
    pending = [arm for arm in SWEEP_ARMS if arm.name not in already_done]
    for name in already_done:
        print(f'skip {name}: checkpoint_{arguments.generation}.json already exists', flush=True)

    if arguments.dry_run:
        for index, arm in enumerate(pending):
            device_id = arguments.device_ids[index % len(arguments.device_ids)]
            print(f'[{arm.name}] device {device_id}: {" ".join(build_command(arm, arguments, device_id))}', flush=True)
        return 0

    free_devices = list(arguments.device_ids)
    queue = list(pending)
    running: list[RunningArm] = []
    failures: list[str] = []

    while queue or running:
        while queue and free_devices:
            arm = queue.pop(0)
            device_id = free_devices.pop(0)
            log_path = arguments.output_root / f'{arm.name}.log'
            handle = log_path.open('w', encoding='utf-8')
            process = subprocess.Popen(
                build_command(arm, arguments, device_id), stdout=handle, stderr=subprocess.STDOUT
            )
            running.append(RunningArm(arm, device_id, process, log_path, time.monotonic()))
            print(f'started {arm.name} on device {device_id}', flush=True)

        time.sleep(POLL_INTERVAL_SECONDS)

        for entry in tuple(running):
            if entry.process.poll() is None:
                continue
            running.remove(entry)
            free_devices.append(entry.device_id)
            minutes = (time.monotonic() - entry.started_at) / 60.0
            if entry.process.returncode:
                failures.append(entry.arm.name)
                print(f'FAILED {entry.arm.name} after {minutes:.1f} min, see {entry.log_path}', flush=True)
            else:
                gap = final_headline_gap(entry.log_path)
                reported = f'{gap:.4f}' if gap is not None else 'unknown'
                print(f'finished {entry.arm.name} in {minutes:.1f} min, headline gap {reported}', flush=True)

    summary = {
        arm.name: final_headline_gap(arguments.output_root / f'{arm.name}.log')
        for arm in SWEEP_ARMS
        if arm.name not in already_done
    }
    write_text_atomically(
        arguments.output_root / 'sweep-summary.json',
        '{\n' + ',\n'.join(f'  "{name}": {value}' for name, value in summary.items()) + '\n}\n',
    )
    print('\nheadline policy gap above floor, lower is better:', flush=True)
    for name, value in sorted(summary.items(), key=lambda item: (item[1] is None, item[1])):
        print(f'  {name:24s} {"failed" if value is None else f"{value:.4f}"}', flush=True)
    return 1 if failures else 0


def parse_arguments() -> SweepArguments:
    parser = argparse.ArgumentParser(description='Train the distillation sweep, one arm per GPU.')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--steps', default=100000, type=int)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--learning-rate', default=0.002, type=float)
    parser.add_argument('--warmup-steps', default=1000, type=int)
    parser.add_argument('--evaluate-every', default=4000, type=int)
    parser.add_argument('--checkpoint-every', default=0, type=int)
    parser.add_argument('--policy-bottleneck-rank', default=16, type=int)
    parser.add_argument('--device-ids', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--random-seed', default=20260827, type=int)
    parser.add_argument('--dry-run', action='store_true')
    namespace = parser.parse_args()

    device_ids = tuple(int(entry) for entry in namespace.device_ids.split(',') if entry.strip())
    if not device_ids:
        raise ValueError('At least one device id is required.')
    if not namespace.dataset.exists():
        raise ValueError(f'Dataset {namespace.dataset} does not exist.')

    return SweepArguments(
        dataset=namespace.dataset,
        output_root=namespace.output_root,
        generation=namespace.generation,
        steps=namespace.steps,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        warmup_steps=namespace.warmup_steps,
        evaluate_every=namespace.evaluate_every,
        checkpoint_every=namespace.checkpoint_every,
        policy_bottleneck_rank=namespace.policy_bottleneck_rank,
        device_ids=device_ids,
        random_seed=namespace.random_seed,
        dry_run=namespace.dry_run,
    )


def main() -> None:
    raise SystemExit(run_sweep(parse_arguments()))


if __name__ == '__main__':
    main()
