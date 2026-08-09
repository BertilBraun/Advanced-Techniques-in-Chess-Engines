from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from src.evaluation.configuration import (
    EvaluationConfiguration,
    FixedCheckpointEvaluationDefinition,
    FixedDatasetEvaluationDefinition,
    KataGoEvaluationDefinition,
    PolicyRandomOpponentEvaluationDefinition,
    PreviousCheckpointEvaluationDefinition,
    RandomOpponentEvaluationDefinition,
    StockfishEvaluationDefinition,
)
from src.evaluation.contracts import (
    CheckpointOpponent,
    EvaluationJob,
    FixedDatasetEvaluationJob,
    KataGoOpponent,
    MatchEvaluationJob,
    RandomOpponent,
    StockfishOpponent,
)
from src.experiment.configuration import ExperimentConfiguration
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.paths import checkpoint_manifest_path
from src.util.frozen_model import FrozenModel


class CheckpointPublication(FrozenModel):
    elapsed_seconds: float = Field(ge=0.0)
    checkpoint: CheckpointReference


class ScheduledEvaluationSuite(FrozenModel):
    boundary_seconds: int = Field(gt=0)
    checkpoint: CheckpointReference


def checkpoint_at(
    publications: tuple[CheckpointPublication, ...],
    boundary_seconds: int,
) -> CheckpointReference:
    available = tuple(publication for publication in publications if publication.elapsed_seconds <= boundary_seconds)
    if not available:
        raise RuntimeError('No complete checkpoint was available at the evaluation boundary.')
    return available[-1].checkpoint


def jobs_for_suite(
    experiment: ExperimentConfiguration,
    run_path: Path,
    result_directory: Path,
    suite: ScheduledEvaluationSuite,
    scheduled_suites: tuple[ScheduledEvaluationSuite, ...],
    next_device_index: int,
) -> tuple[tuple[EvaluationJob, ...], int]:
    jobs: list[EvaluationJob] = []
    configuration = experiment.evaluation
    device_cycle = experiment.training.topology.evaluation.device_cycle
    device_index = next_device_index
    for definition in configuration.definitions:
        opponent = None
        match definition:
            case FixedDatasetEvaluationDefinition():
                kind = 'fixed_dataset'
            case RandomOpponentEvaluationDefinition() | PolicyRandomOpponentEvaluationDefinition():
                kind = 'match'
                opponent = RandomOpponent(kind='random')
            case PreviousCheckpointEvaluationDefinition(boundary_offset=offset):
                opponent_boundary = suite.boundary_seconds - offset * configuration.cadence_seconds
                previous = next(
                    (
                        scheduled.checkpoint
                        for scheduled in scheduled_suites
                        if scheduled.boundary_seconds == opponent_boundary
                    ),
                    None,
                )
                if previous is None or previous.generation >= suite.checkpoint.generation:
                    continue
                kind = 'match'
                opponent = CheckpointOpponent(kind='checkpoint', checkpoint=previous)
            case FixedCheckpointEvaluationDefinition(generation=generation):
                if (
                    generation >= suite.checkpoint.generation
                    or not checkpoint_manifest_path(generation, run_path).is_file()
                ):
                    continue
                kind = 'match'
                opponent = CheckpointOpponent(
                    kind='checkpoint',
                    checkpoint=CheckpointReference.load_for_inference(run_path, generation),
                )
            case StockfishEvaluationDefinition(skill_level=skill_level):
                kind = 'match'
                opponent = StockfishOpponent(kind='stockfish', skill_level=skill_level)
            case KataGoEvaluationDefinition():
                kind = 'match'
                opponent = KataGoOpponent(kind='katago')
        device_id = device_cycle[device_index % len(device_cycle)]
        device_index += 1
        job_id = f'{suite.boundary_seconds:010d}-{definition.definition_id}-g{suite.checkpoint.generation}'
        common = {
            'job_id': job_id,
            'definition': definition,
            'boundary_seconds': suite.boundary_seconds,
            'candidate': suite.checkpoint,
            'device_id': device_id,
            'deadline_seconds': configuration.job_timeout_seconds,
            'random_seed': _job_seed(
                experiment.training.random_seed,
                suite.boundary_seconds,
                definition.definition_id,
            ),
            'result_path': result_directory / f'{job_id}.json',
        }
        if kind == 'fixed_dataset':
            jobs.append(FixedDatasetEvaluationJob(kind='fixed_dataset', **common))
        else:
            assert opponent is not None
            jobs.append(MatchEvaluationJob(kind='match', opponent=opponent, **common))
    return tuple(jobs), device_index


def required_checkpoint_generations(
    configuration: EvaluationConfiguration,
    scheduled_suites: tuple[ScheduledEvaluationSuite, ...],
    pending_jobs: tuple[EvaluationJob, ...],
) -> tuple[int, ...]:
    fixed_generations: set[int] = set()
    maximum_previous_offset = 0
    for definition in configuration.definitions:
        match definition:
            case FixedCheckpointEvaluationDefinition(generation=generation):
                fixed_generations.add(generation)
            case PreviousCheckpointEvaluationDefinition(boundary_offset=offset):
                maximum_previous_offset = max(maximum_previous_offset, offset)
            case _:
                pass
    recent_suites = scheduled_suites[-maximum_previous_offset:] if maximum_previous_offset else ()
    required = fixed_generations | {suite.checkpoint.generation for suite in recent_suites}
    for job in pending_jobs:
        required.add(job.candidate.generation)
        match job:
            case MatchEvaluationJob(opponent=CheckpointOpponent(checkpoint=checkpoint)):
                required.add(checkpoint.generation)
            case FixedDatasetEvaluationJob() | MatchEvaluationJob():
                pass
    return tuple(sorted(required))


def _job_seed(run_seed: int, boundary_seconds: int, definition_id: str) -> int:
    payload = f'{run_seed}:{boundary_seconds}:{definition_id}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
