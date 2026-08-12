from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time

from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.self_play.protocol import PausedSelfPlayState, RunningSelfPlayState, StatisticsLevel
from src.training.checkpoint import CheckpointReference
from src.training.self_play_group import SelfPlayGroup


@dataclass(frozen=True)
class Arguments:
    run_config: Path
    run_directory: Path
    source_generation: int
    target_generation: int
    settle_seconds: float
    output: Path


@dataclass(frozen=True)
class TransitionBenchmarkResult:
    worker_count: int
    paused_worker_ids: tuple[int, ...]
    active_worker_ids: tuple[int, ...]
    initial_checkpoint_application_seconds: float
    pause_acknowledgement_seconds: float
    target_checkpoint_application_seconds: float


def _elapsed_apply(group: SelfPlayGroup, desired_states: tuple[RunningSelfPlayState, ...]) -> float:
    started_at = time.perf_counter()
    responses = group.apply(desired_states)
    elapsed_seconds = time.perf_counter() - started_at
    if any(response.kind != 'running' for response in responses):
        raise RuntimeError('Self-play workers did not enter the running state.')
    return elapsed_seconds


def run_benchmark(arguments: Arguments) -> TransitionBenchmarkResult:
    experiment = load_experiment_configuration(arguments.run_config)
    source_checkpoint = CheckpointReference.load_for_inference(arguments.run_directory, arguments.source_generation)
    target_checkpoint = CheckpointReference.load_for_inference(arguments.run_directory, arguments.target_generation)
    paused_worker_ids = experiment.training.topology.self_play.node_ids_to_pause_during_training
    active_worker_ids = tuple(
        worker_id
        for worker_id in range(len(experiment.training.topology.self_play.device_ids))
        if worker_id not in paused_worker_ids
    )
    with tempfile.TemporaryDirectory(prefix='self-play-transition-benchmark-') as temporary_directory:
        training = experiment.training.validated_copy(update={'save_path': temporary_directory})
        experiment = experiment.validated_copy(update={'training': training.model_dump(mode='json')})
        group = SelfPlayGroup(create_game_implementation(experiment))
        try:
            initial_seconds = _elapsed_apply(
                group,
                tuple(RunningSelfPlayState(checkpoint=source_checkpoint) for _ in range(group.worker_count)),
            )
            time.sleep(arguments.settle_seconds)
            started_at = time.perf_counter()
            paused = group.apply_to_workers(paused_worker_ids, PausedSelfPlayState())
            pause_seconds = time.perf_counter() - started_at
            if any(response.kind != 'paused' for response in paused):
                raise RuntimeError('Selected self-play workers did not enter the paused state.')
            target_seconds = _elapsed_apply(
                group,
                tuple(
                    RunningSelfPlayState(
                        checkpoint=target_checkpoint,
                        completed_generation_statistics=StatisticsLevel.BASIC,
                    )
                    for _ in range(group.worker_count)
                ),
            )
        finally:
            group.close()
    return TransitionBenchmarkResult(
        worker_count=group.worker_count,
        paused_worker_ids=paused_worker_ids,
        active_worker_ids=active_worker_ids,
        initial_checkpoint_application_seconds=initial_seconds,
        pause_acknowledgement_seconds=pause_seconds,
        target_checkpoint_application_seconds=target_seconds,
    )


def _parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Benchmark self-play pause and checkpoint acknowledgement latency.')
    parser.add_argument('--run-config', type=Path, required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--source-generation', type=int, required=True)
    parser.add_argument('--target-generation', type=int, required=True)
    parser.add_argument('--settle-seconds', type=float, default=10.0)
    parser.add_argument('--output', type=Path, required=True)
    namespace = parser.parse_args()
    return Arguments(
        run_config=namespace.run_config,
        run_directory=namespace.run_directory,
        source_generation=namespace.source_generation,
        target_generation=namespace.target_generation,
        settle_seconds=namespace.settle_seconds,
        output=namespace.output,
    )


def main() -> None:
    arguments = _parse_arguments()
    result = run_benchmark(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(asdict(result), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(asdict(result), indent=2))


if __name__ == '__main__':
    main()
