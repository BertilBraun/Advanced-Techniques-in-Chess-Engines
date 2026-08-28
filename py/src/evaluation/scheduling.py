from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from src.evaluation.configuration import (
    EvaluationConfiguration,
    EvaluationDefinition,
    FixedDatasetEvaluationDefinition,
    KataGoEvaluationDefinition,
    PairedMatchEvaluationDefinition,
    PolicyRandomOpponentEvaluationDefinition,
    PreviousCheckpointEvaluationDefinition,
    RandomOpponentEvaluationDefinition,
    ReferenceCheckpointEvaluationDefinition,
    StockfishEvaluationDefinition,
    StockfishFixedNodesEvaluationDefinition,
)
from src.evaluation.contracts import (
    CheckpointOpponent,
    ElapsedCheckpointReference,
    EvaluationJob,
    EvaluationOpponent,
    EvaluationReferenceManifest,
    FixedDatasetEvaluationJob,
    KataGoOpponent,
    MatchEvaluationJob,
    RandomOpponent,
    StockfishFixedNodesOpponent,
    StockfishOpponent,
)
from src.experiment.configuration import ExperimentConfiguration
from src.training.checkpoint import CheckpointReference
from src.util.frozen_model import FrozenModel


class CheckpointPublication(FrozenModel):
    elapsed_seconds: float = Field(ge=0.0)
    checkpoint: CheckpointReference


ScheduledEvaluationSuite = ElapsedCheckpointReference


@dataclass(frozen=True)
class _EvaluationJobContext:
    job_id: str
    boundary_seconds: int
    candidate: CheckpointReference
    device_id: int
    deadline_seconds: float
    random_seed: int
    result_path: Path


def checkpoint_at(
    publications: tuple[CheckpointPublication, ...],
    boundary_seconds: int,
) -> CheckpointReference:
    available = tuple(publication for publication in publications if publication.elapsed_seconds <= boundary_seconds)
    if not available:
        raise RuntimeError('No complete checkpoint was available at the evaluation boundary.')
    return available[-1].checkpoint


def _previous_checkpoint(
    definition: PreviousCheckpointEvaluationDefinition,
    configuration: EvaluationConfiguration,
    suite: ScheduledEvaluationSuite,
    scheduled_suites: tuple[ScheduledEvaluationSuite, ...],
) -> CheckpointReference | None:
    opponent_boundary = suite.boundary_seconds - definition.boundary_offset * configuration.cadence_seconds
    previous = next(
        (scheduled.checkpoint for scheduled in scheduled_suites if scheduled.boundary_seconds == opponent_boundary),
        None,
    )
    if previous is None or previous.generation >= suite.checkpoint.generation:
        return None
    return previous


def _reference_checkpoint(
    definition: ReferenceCheckpointEvaluationDefinition,
    suite: ScheduledEvaluationSuite,
) -> CheckpointReference | None:
    manifest_path = Path(definition.manifest_path)
    if not manifest_path.is_file():
        return None
    manifest = EvaluationReferenceManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    reference = next(
        (entry.checkpoint for entry in manifest.checkpoints if entry.boundary_seconds == suite.boundary_seconds),
        None,
    )
    if reference is None or reference.inference_model_sha256 == suite.checkpoint.inference_model_sha256:
        return None
    reference.validate_inference_model()
    return reference


def _match_job(
    definition: RandomOpponentEvaluationDefinition
    | PolicyRandomOpponentEvaluationDefinition
    | PreviousCheckpointEvaluationDefinition
    | ReferenceCheckpointEvaluationDefinition
    | StockfishEvaluationDefinition
    | StockfishFixedNodesEvaluationDefinition
    | KataGoEvaluationDefinition,
    opponent: EvaluationOpponent,
    context: _EvaluationJobContext,
) -> MatchEvaluationJob:
    return MatchEvaluationJob(
        kind='match',
        job_id=context.job_id,
        definition=definition,
        boundary_seconds=context.boundary_seconds,
        candidate=context.candidate,
        opponent=opponent,
        device_id=context.device_id,
        deadline_seconds=context.deadline_seconds,
        random_seed=context.random_seed,
        result_path=context.result_path,
    )


def _job_for_definition(
    definition: EvaluationDefinition,
    configuration: EvaluationConfiguration,
    suite: ScheduledEvaluationSuite,
    scheduled_suites: tuple[ScheduledEvaluationSuite, ...],
    context: _EvaluationJobContext,
) -> EvaluationJob | None:
    match definition:
        case FixedDatasetEvaluationDefinition():
            return FixedDatasetEvaluationJob(
                kind='fixed_dataset',
                job_id=context.job_id,
                definition=definition,
                boundary_seconds=context.boundary_seconds,
                candidate=context.candidate,
                device_id=context.device_id,
                deadline_seconds=context.deadline_seconds,
                random_seed=context.random_seed,
                result_path=context.result_path,
            )
        case RandomOpponentEvaluationDefinition() | PolicyRandomOpponentEvaluationDefinition():
            return _match_job(definition, RandomOpponent(kind='random'), context)
        case PreviousCheckpointEvaluationDefinition():
            boundary_index = suite.boundary_seconds // configuration.cadence_seconds
            if definition.boundary_parity != 'every' and definition.boundary_parity != (
                'even' if boundary_index % 2 == 0 else 'odd'
            ):
                return None
            checkpoint = _previous_checkpoint(definition, configuration, suite, scheduled_suites)
            return (
                None
                if checkpoint is None
                else _match_job(
                    definition,
                    CheckpointOpponent(kind='checkpoint', checkpoint=checkpoint),
                    context,
                )
            )
        case ReferenceCheckpointEvaluationDefinition():
            checkpoint = _reference_checkpoint(definition, suite)
            return (
                None
                if checkpoint is None
                else _match_job(
                    definition,
                    CheckpointOpponent(kind='checkpoint', checkpoint=checkpoint),
                    context,
                )
            )
        case StockfishEvaluationDefinition(skill_level=skill_level):
            return _match_job(definition, StockfishOpponent(kind='stockfish', skill_level=skill_level), context)
        case StockfishFixedNodesEvaluationDefinition(nodes=nodes, engine_executable_path=engine_executable_path):
            return _match_job(
                definition,
                StockfishFixedNodesOpponent(
                    kind='stockfish_fixed_nodes',
                    nodes=nodes,
                    engine_executable_path=engine_executable_path,
                ),
                context,
            )
        case KataGoEvaluationDefinition(maximum_visits=maximum_visits):
            return _match_job(
                definition,
                KataGoOpponent(kind='katago', maximum_visits=maximum_visits),
                context,
            )


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
        if isinstance(definition, PairedMatchEvaluationDefinition) and not definition.is_active_at(
            suite.checkpoint.generation
        ):
            continue
        device_id = device_cycle[device_index % len(device_cycle)]
        job_id = f'{suite.boundary_seconds:010d}-{definition.definition_id}-g{suite.checkpoint.generation}'
        context = _EvaluationJobContext(
            job_id=job_id,
            boundary_seconds=suite.boundary_seconds,
            candidate=suite.checkpoint,
            device_id=device_id,
            deadline_seconds=configuration.job_timeout_seconds,
            random_seed=_job_seed(
                experiment.training.random_seed,
                suite.boundary_seconds,
                definition.definition_id,
            ),
            result_path=result_directory / f'{job_id}.json',
        )
        job = _job_for_definition(definition, configuration, suite, scheduled_suites, context)
        if job is None:
            continue
        jobs.append(job)
        device_index += 1
    return tuple(jobs), device_index


def required_checkpoint_generations(
    configuration: EvaluationConfiguration,
    scheduled_suites: tuple[ScheduledEvaluationSuite, ...],
    pending_jobs: tuple[EvaluationJob, ...],
) -> tuple[int, ...]:
    maximum_previous_offset = 0
    for definition in configuration.definitions:
        match definition:
            case PreviousCheckpointEvaluationDefinition(boundary_offset=offset):
                maximum_previous_offset = max(maximum_previous_offset, offset)
            case _:
                pass
    recent_suites = scheduled_suites[-maximum_previous_offset:] if maximum_previous_offset else ()
    required = {suite.checkpoint.generation for suite in scheduled_suites} | {
        suite.checkpoint.generation for suite in recent_suites
    }
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
