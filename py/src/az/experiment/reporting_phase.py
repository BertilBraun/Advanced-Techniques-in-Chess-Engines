from __future__ import annotations

from uuid import uuid5

import torch

from src.az.calibration.models import load_trace_collection_artifact
from src.az.config.manifest import RunManifest
from src.az.config.model import FixedModelSchedule
from src.az.config.serialization import load_resolved_configuration, model_sha256
from src.az.evaluation.models import EvaluationGameResult
from src.az.evaluation.protocol import derive_evaluation_id
from src.az.experiment.commit_journal import ReplayCommitJournal
from src.az.experiment.evaluation_phase import (
    evaluation_pair,
    load_checkpoint_claims,
    load_evaluation_opponent,
)
from src.az.experiment.lifecycle import (
    ExperimentPhase,
    ExperimentRunRepository,
    ExperimentRunState,
    RunArtifactKind,
    require_exact_artifact_files,
)
from src.az.experiment.phase_support import (
    begin_phase,
    complete_phase,
    registered_artifact_paths,
)
from src.az.reporting.build import (
    EvaluationCheckpointEvidence,
    RunReportEvidence,
    build_report,
)
from src.az.reporting.models import CheckpointTimingEvidence, RunIdentity
from src.az.reporting.render import render_csv, render_machine_json, render_markdown
from src.az.training.checkpoints import (
    CheckpointPointer,
    CheckpointRepository,
    TrainerCheckpointState,
)


def run_reporting(repository: ExperimentRunRepository) -> ExperimentRunState:
    state = begin_phase(repository, ExperimentPhase.REPORTING)
    configuration = load_resolved_configuration(repository.configuration_path)
    manifest = RunManifest.model_validate_json((repository.directory / repository.MANIFEST_FILENAME).read_bytes())
    match configuration.model.schedule:
        case FixedModelSchedule(architecture=architecture):
            pass
        case _:
            raise ValueError('The current Go report pipeline requires a fixed model schedule.')
    commit_journal = ReplayCommitJournal((repository.directory / 'replay-commits.azc').resolve())
    checkpoint_repository = CheckpointRepository(
        (repository.directory / 'checkpoints').resolve(),
        state.run_id,
        state.resolved_configuration_sha256,
    )
    result_paths = registered_artifact_paths(
        repository,
        state,
        RunArtifactKind.EVALUATION_RESULT,
    )
    if not result_paths:
        raise ValueError('Reporting requires completed evaluation results.')
    require_exact_artifact_files(
        repository.directory / 'evaluation-results',
        '*.json',
        result_paths,
    )
    loaded_games = tuple(EvaluationGameResult.model_validate_json(path.read_bytes()) for path in result_paths)
    claims = load_checkpoint_claims(repository, state)
    opponent, _ = load_evaluation_opponent(
        configuration,
        configuration.game,
        architecture,
        torch.device('cpu'),
        repository.directory,
    )
    search_sha256 = model_sha256(configuration.evaluation.search)
    evaluation_checkpoints: list[EvaluationCheckpointEvidence] = []
    checkpoint_timing: list[CheckpointTimingEvidence] = []
    for evaluation_index, claim in enumerate(claims):
        evaluation_id = derive_evaluation_id(
            state.run_id,
            state.resolved_configuration_sha256,
            search_sha256,
            evaluation_index,
            claim.requested_elapsed_seconds,
            claim.candidate,
            opponent,
            configuration.game,
        )
        games = tuple(game for game in loaded_games if game.evaluation_id == evaluation_id)
        pair_indices = tuple(sorted({game.pair_index for game in games}))
        if len(pair_indices) != (configuration.evaluation.paired_games_per_checkpoint // 2):
            raise ValueError('Reporting requires every scheduled evaluation pair.')
        pairs = tuple(evaluation_pair(evaluation_id, pair_index, games) for pair_index in pair_indices)
        evaluation_checkpoints.append(
            EvaluationCheckpointEvidence(
                elapsed_hours=claim.requested_elapsed_seconds / 3600,
                pairs=pairs,
                bootstrap_samples=configuration.evaluation.bootstrap_samples,
                confidence_level=configuration.evaluation.confidence_level,
                bootstrap_seed=configuration.evaluation.bootstrap_seed,
            )
        )
        checkpoint_timing.append(
            CheckpointTimingEvidence(
                requested_elapsed_seconds=claim.requested_elapsed_seconds,
                published_elapsed_seconds=claim.published_elapsed_seconds,
                checkpoint_id=claim.candidate.checkpoint_id,
                model_artifact_sha256=claim.candidate.model_artifact_sha256,
            )
        )
    trainer_state = current_trainer_state(
        checkpoint_repository,
        len(configuration.topology.trainer.device_ids),
    )
    trace_paths = registered_artifact_paths(
        repository,
        state,
        RunArtifactKind.SEARCH_TRACE,
    )
    require_exact_artifact_files(
        repository.directory / 'search-traces',
        'trace-*.json',
        trace_paths,
    )
    evidence = RunReportEvidence(
        identity=RunIdentity(
            run_id=state.run_id,
            arm_id=uuid5(state.run_id, f'arm:{configuration.experiment.arm_id}'),
            seed=configuration.experiment.root_seed,
            resolved_configuration_sha256=state.resolved_configuration_sha256,
            source_revision=state.source_revision,
            hardware_identity=model_sha256(manifest.hardware),
        ),
        committed_replay_envelopes=commit_journal.envelopes,
        evaluation_checkpoints=tuple(evaluation_checkpoints),
        checkpoint_timing=tuple(checkpoint_timing),
        optimizer_steps=trainer_state.replay_credits.completed_optimizer_steps,
        replay_reuse=float(
            trainer_state.replay_credits.consumed_position_credits
            / max(
                1,
                trainer_state.replay_credits.credited_unique_positions,
            )
        ),
        gpu_utilization_percent=None,
        source_artifact_sha256s=tuple(artifact.sha256 for artifact in state.artifacts),
        search_trace_artifacts=tuple(load_trace_collection_artifact(path).artifact for path in trace_paths),
    )
    report = build_report(
        report_id=uuid5(state.run_id, 'report'),
        title=f'{configuration.experiment.name}: {configuration.experiment.arm_id}',
        matrix_id=uuid5(state.run_id, 'single-run-matrix'),
        common_controls_sha256=model_sha256(configuration.evaluation.search),
        runs=(evidence,),
    )
    report_directory = repository.directory / 'report'
    report_directory.mkdir(exist_ok=True)
    outputs = (
        (report_directory / 'report.json', render_machine_json(report.payload)),
        (report_directory / 'report.md', render_markdown(report.payload)),
        (report_directory / 'report.csv', render_csv(report.payload)),
    )
    for path, contents in outputs:
        path.write_text(contents, encoding='utf-8', newline='\n')
    artifacts = tuple(repository.artifact(RunArtifactKind.RESEARCH_REPORT, path) for path, _ in outputs)
    return complete_phase(
        repository,
        state,
        ExperimentPhase.REPORTING,
        artifacts,
    )


def current_trainer_state(
    repository: CheckpointRepository,
    configured_world_size: int,
) -> TrainerCheckpointState:
    pointer = CheckpointPointer.model_validate_json(repository.pointer_path.read_bytes())
    if pointer.checkpoint_directory.startswith('distributed-'):
        states = tuple(repository.load_distributed(rank).rank.state for rank in range(configured_world_size))
        if len(set(states)) != 1:
            raise ValueError('Distributed checkpoint ranks disagree on trainer state.')
        return states[0]
    return repository.load_current().manifest.state
