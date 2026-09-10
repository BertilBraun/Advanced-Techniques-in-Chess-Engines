from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field
from src.training.checkpoint import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from tools.distill_match import DistillationMatchResult
from tools.v34_orchestration import ChildCommand, parse_devices, run_child_commands, validate_input_paths

HEADLINE_PATTERN = re.compile(r'headline policy gap above floor ([0-9.]+)')


@dataclass(frozen=True)
class Arguments:
    teacher_run_state: Path
    teacher_generation: int
    replay_store: Path
    experiment: Path
    opening_manifest: Path
    output_root: Path
    devices: tuple[int, ...]
    seeds: tuple[int, ...]
    student_generation: int
    steps: int
    searches_per_move: int
    parallel_searches: int
    throughput_device: int
    dry_run: bool


@dataclass(frozen=True)
class Architecture:
    name: str
    layers: int
    hidden_size: int


@dataclass(frozen=True)
class TrainingArm:
    architecture: Architecture
    random_seed: int
    device_id: int

    @property
    def name(self) -> str:
        return f'{self.architecture.name}-seed-{self.random_seed}'


class ArmMetric(FrozenModel):
    architecture: str
    random_seed: int = Field(ge=0)
    headline_policy_gap: float = Field(ge=0.0)
    run_state: Path


class DistillationSelection(FrozenModel):
    schema_version: Literal[1] = 1
    metrics: tuple[ArmMetric, ...]
    selected_architecture: str
    selected_run_state: Path
    selected_generation: int = Field(ge=0)
    selection_rule: Literal['lowest_mean_policy_gap_above_floor'] = 'lowest_mean_policy_gap_above_floor'


ARCHITECTURES = (
    Architecture('4x80', 4, 80),
    Architecture('5x72', 5, 72),
    Architecture('6x64', 6, 64),
    Architecture('8x56', 8, 56),
)


def training_arms(arguments: Arguments) -> tuple[TrainingArm, ...]:
    candidates = tuple((architecture, random_seed) for architecture in ARCHITECTURES for random_seed in arguments.seeds)
    if len(arguments.devices) < len(candidates):
        raise ValueError(f'The {len(candidates)} training arms require at least that many devices.')
    return tuple(
        TrainingArm(architecture, random_seed, arguments.devices[index])
        for index, (architecture, random_seed) in enumerate(candidates)
    )


def _training_command(arguments: Arguments, arm: TrainingArm, replay_store_sha256: str) -> tuple[str, ...]:
    return (
        sys.executable,
        '-m',
        'tools.distill_train_student',
        '--replay-store',
        str(arguments.replay_store),
        '--replay-experiment',
        str(arguments.experiment),
        '--replay-store-sha256',
        replay_store_sha256,
        '--output-run-state',
        str(arguments.output_root / 'students' / arm.name),
        '--network-kind',
        'convolutional',
        '--layers',
        str(arm.architecture.layers),
        '--hidden-size',
        str(arm.architecture.hidden_size),
        '--policy-head-kind',
        'from_to_attention',
        '--policy-key-size',
        '64',
        '--batch-size',
        '1024',
        '--steps',
        str(arguments.steps),
        '--learning-rate',
        '0.002',
        '--learning-rate-schedule',
        'plateau',
        '--anneal-fraction',
        '0.2',
        '--warmup-steps',
        '1000',
        '--holdout-fraction',
        '0.02',
        '--evaluate-every',
        '4000',
        '--checkpoint-every',
        '10000',
        '--device-id',
        str(arm.device_id),
        '--random-seed',
        str(arm.random_seed),
        '--generation',
        str(arguments.student_generation),
    )


def _arm_is_complete(run_state: Path, log_path: Path, generation: int) -> bool:
    checkpoint_path = run_state / f'checkpoint_{generation}.json'
    if not checkpoint_path.exists() and not log_path.exists() and not run_state.exists():
        return False
    if not checkpoint_path.is_file() or not log_path.is_file():
        raise ValueError(f'Incomplete existing training evidence in {run_state}; preserve it before retrying.')
    checkpoint = CheckpointReference.load_for_inference(run_state, generation)
    checkpoint.validate_inference_model()
    _headline_gap(log_path)
    return True


def training_commands(arguments: Arguments, replay_store_sha256: str) -> tuple[ChildCommand, ...]:
    commands: list[ChildCommand] = []
    for arm in training_arms(arguments):
        run_state = arguments.output_root / 'students' / arm.name
        log_path = arguments.output_root / 'logs' / f'{arm.name}.log'
        if _arm_is_complete(run_state, log_path, arguments.student_generation):
            continue
        commands.append(ChildCommand(arm.name, _training_command(arguments, arm, replay_store_sha256), log_path))
    return tuple(commands)


def _headline_gap(log_path: Path) -> float:
    matches = HEADLINE_PATTERN.findall(log_path.read_text(encoding='utf-8', errors='strict'))
    if not matches:
        raise ValueError(f'Training log has no final policy-gap measurement: {log_path}')
    return float(matches[-1])


def select_student(arguments: Arguments) -> DistillationSelection:
    metrics = tuple(
        ArmMetric(
            architecture=arm.architecture.name,
            random_seed=arm.random_seed,
            headline_policy_gap=_headline_gap(arguments.output_root / 'logs' / f'{arm.name}.log'),
            run_state=(arguments.output_root / 'students' / arm.name).resolve(),
        )
        for arm in training_arms(arguments)
    )
    means = tuple(
        (
            architecture.name,
            sum(metric.headline_policy_gap for metric in metrics if metric.architecture == architecture.name)
            / len(arguments.seeds),
        )
        for architecture in ARCHITECTURES
    )
    selected_name = min(means, key=lambda item: item[1])[0]
    selected_metric = min(
        (metric for metric in metrics if metric.architecture == selected_name),
        key=lambda metric: metric.headline_policy_gap,
    )
    return DistillationSelection(
        metrics=metrics,
        selected_architecture=selected_name,
        selected_run_state=selected_metric.run_state,
        selected_generation=arguments.student_generation,
    )


def _match_command(
    arguments: Arguments,
    selection: DistillationSelection,
    mode: str,
    output: Path,
    pinned_ratio: float | None = None,
) -> tuple[str, ...]:
    command = (
        sys.executable,
        '-m',
        'tools.distill_match',
        '--teacher-run-state',
        str(arguments.teacher_run_state),
        '--teacher-generation',
        str(arguments.teacher_generation),
        '--student-run-state',
        str(selection.selected_run_state),
        '--student-generation',
        str(selection.selected_generation),
        '--openings-manifest',
        str(arguments.opening_manifest),
        '--experiment-config',
        str(arguments.experiment),
        '--mode',
        mode,
        '--searches-per-move',
        str(arguments.searches_per_move),
        '--parallel-searches',
        str(arguments.parallel_searches),
        '--exploration-constant',
        '1.0',
        '--opening-pair-count',
        '100',
        '--throughput-position-count',
        '100',
        '--device-id',
        str(arguments.throughput_device),
        '--random-seed',
        '20260816',
        '--output',
        str(output),
    )
    if pinned_ratio is None:
        return command
    return (*command, '--pinned-throughput-ratio', str(pinned_ratio))


def _run_stage(command: ChildCommand, output: Path, dry_run: bool) -> None:
    if output.is_file():
        DistillationMatchResult.model_validate_json(output.read_text(encoding='utf-8'))
        return
    outcomes = run_child_commands((command,), dry_run)
    if outcomes and outcomes[0].return_code != 0:
        raise RuntimeError(f'{command.name} failed; see {command.log_path}.')


def run_distillation(arguments: Arguments) -> int:
    if arguments.dry_run:
        run_child_commands(training_commands(arguments, '<replay-store-sha256>'), True)
        return 0
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    replay_store_sha256 = file_sha256(arguments.replay_store)
    replay_hash_path = arguments.output_root / 'replay-store.sha256'
    if replay_hash_path.exists() and replay_hash_path.read_text(encoding='utf-8').strip() != replay_store_sha256:
        raise ValueError('The replay store hash differs from the existing distillation evidence.')
    if not replay_hash_path.exists():
        write_text_atomically(replay_hash_path, replay_store_sha256 + '\n')
    commands = training_commands(arguments, replay_store_sha256)
    if commands:
        outcomes = run_child_commands(commands, arguments.dry_run)
        if any(outcome.return_code != 0 for outcome in outcomes):
            return 1
    selection = select_student(arguments)
    selection_path = arguments.output_root / 'selection.json'
    write_text_atomically(selection_path, selection.model_dump_json(indent=2) + '\n')
    evaluation_root = arguments.output_root / 'evaluation'
    throughput_path = evaluation_root / 'throughput.json'
    _run_stage(
        ChildCommand(
            'throughput',
            _match_command(arguments, selection, 'throughput-only', throughput_path),
            arguments.output_root / 'logs' / 'throughput.log',
        ),
        throughput_path,
        False,
    )
    throughput = DistillationMatchResult.model_validate_json(throughput_path.read_text(encoding='utf-8'))
    if throughput.throughput_ratio is None:
        raise RuntimeError('Throughput result did not record a student/teacher search-throughput ratio.')

    equal_searches_path = evaluation_root / 'equal-searches.json'
    _run_stage(
        ChildCommand(
            'equal-searches',
            _match_command(arguments, selection, 'equal-nodes', equal_searches_path),
            arguments.output_root / 'logs' / 'equal-searches.log',
        ),
        equal_searches_path,
        False,
    )
    equal_time_path = evaluation_root / 'equal-expected-time.json'
    _run_stage(
        ChildCommand(
            'equal-expected-time',
            _match_command(
                arguments,
                selection,
                'equal-compute',
                equal_time_path,
                pinned_ratio=throughput.throughput_ratio,
            ),
            arguments.output_root / 'logs' / 'equal-expected-time.log',
        ),
        equal_time_path,
        False,
    )
    return 0


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Train, select, and evaluate the v34 replay-compression student.')
    parser.add_argument('--teacher-run-state', required=True, type=Path)
    parser.add_argument('--teacher-generation', required=True, type=int)
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--opening-manifest', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--devices', nargs='+', default=tuple(range(8)), type=int)
    parser.add_argument('--seeds', nargs='+', default=(20260827, 20260828), type=int)
    parser.add_argument('--student-generation', default=0, type=int)
    parser.add_argument('--steps', default=100000, type=int)
    parser.add_argument('--searches-per-move', required=True, type=int)
    parser.add_argument('--parallel-searches', required=True, type=int)
    parser.add_argument('--throughput-device', required=True, type=int)
    parser.add_argument('--dry-run', action='store_true')
    namespace = parser.parse_args()
    arguments = Arguments(
        teacher_run_state=namespace.teacher_run_state,
        teacher_generation=namespace.teacher_generation,
        replay_store=namespace.replay_store,
        experiment=namespace.experiment,
        opening_manifest=namespace.opening_manifest,
        output_root=namespace.output_root,
        devices=parse_devices(tuple(namespace.devices), 'Student sweep'),
        seeds=tuple(namespace.seeds),
        student_generation=namespace.student_generation,
        steps=namespace.steps,
        searches_per_move=namespace.searches_per_move,
        parallel_searches=namespace.parallel_searches,
        throughput_device=namespace.throughput_device,
        dry_run=namespace.dry_run,
    )
    validate_input_paths(
        (arguments.teacher_run_state, arguments.replay_store, arguments.experiment, arguments.opening_manifest)
    )
    if len(arguments.seeds) != 2 or len(set(arguments.seeds)) != 2 or min(arguments.seeds) < 0:
        raise ValueError('The architecture sweep requires exactly two distinct nonnegative seeds.')
    if (
        min(
            arguments.teacher_generation,
            arguments.student_generation,
            arguments.steps,
            arguments.searches_per_move,
            arguments.parallel_searches,
            arguments.throughput_device,
        )
        < 0
    ):
        raise ValueError('Generations, steps, searches, parallelism, and device IDs must be nonnegative.')
    if min(arguments.steps, arguments.searches_per_move, arguments.parallel_searches) == 0:
        raise ValueError('Steps, searches, and parallel searches must be positive.')
    if arguments.searches_per_move <= arguments.parallel_searches:
        raise ValueError('Searches per move must exceed parallel searches.')
    training_arms(arguments)
    return arguments


def main() -> None:
    raise SystemExit(run_distillation(parse_arguments()))


if __name__ == '__main__':
    main()
