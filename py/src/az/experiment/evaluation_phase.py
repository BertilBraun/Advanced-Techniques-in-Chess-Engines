from __future__ import annotations

import hashlib
import io
from contextlib import ExitStack
from pathlib import Path
from uuid import UUID

import torch

from src.az.config.model import FixedModelSchedule
from src.az.config.root import ResolvedRunConfiguration
from src.az.config.serialization import load_resolved_configuration, model_sha256
from src.az.evaluation.checkpoints import EvaluationModelArtifactRepository
from src.az.evaluation.models import (
    CandidateCheckpointIdentity,
    CheckpointOpponentIdentity,
    EvaluationGameResult,
    EvaluationOpponentIdentity,
    EvaluationPairResult,
    RandomOpponentIdentity,
)
from src.az.evaluation.protocol import (
    LoadedEvaluationModel,
    NativeCheckpointEvaluationPlayer,
    PairedEvaluationSpecification,
    PairedGoEvaluator,
    RandomGoEvaluationPlayer,
    derive_evaluation_id,
)
from src.az.evaluation.storage import EvaluationResultRepository
from src.az.experiment.lifecycle import (
    ExperimentPhase,
    ExperimentRunRepository,
    ExperimentRunState,
    RunArtifactKind,
    require_exact_artifact_files,
)
from src.az.experiment.phase_support import (
    ScheduledCheckpointClaim,
    begin_phase,
    complete_phase,
    interrupt_phase,
    registered_artifact_paths,
)
from src.az.games.go.configuration import (
    CheckpointGoOpponent,
    GoGameConfiguration,
    ResidualGoModelConfiguration,
    RandomGoOpponent,
)
from src.az.games.go.model import ResidualGoModel
from src.az.inference.go_batching import GoInferenceBatchBroker
from src.az.training.checkpoints import ModelCheckpointManifest


def run_evaluation(repository: ExperimentRunRepository) -> ExperimentRunState:
    state = begin_phase(repository, ExperimentPhase.EVALUATION)
    configuration = load_resolved_configuration(repository.configuration_path)
    match configuration.game:
        case GoGameConfiguration() as game:
            pass
        case _:
            raise ValueError('Go evaluation requires a Go game configuration.')
    match configuration.model.schedule:
        case FixedModelSchedule(architecture=architecture):
            pass
        case _:
            raise ValueError('The current Go evaluator requires a fixed model schedule.')
    model_artifacts = EvaluationModelArtifactRepository((repository.directory / 'evaluation-models').resolve())
    evaluation_device = (
        torch.device('cpu')
        if configuration.hardware.profile_name == 'local-cpu-smoke'
        else torch.device(f'cuda:{configuration.topology.evaluation.device_ids[0]}')
    )
    if evaluation_device.type == 'cpu':
        torch.set_num_threads(1)
    claims = load_checkpoint_claims(repository, state)
    require_exact_artifact_files(
        repository.directory / 'evaluation-models',
        '*.pt',
        tuple(model_artifacts.path(claim.candidate) for claim in claims),
    )
    search_sha256 = model_sha256(configuration.evaluation.search)
    opponent, opponent_model = load_evaluation_opponent(
        configuration,
        game,
        architecture,
        evaluation_device,
        repository.directory,
    )
    result_repository = EvaluationResultRepository((repository.directory / 'evaluation-results').resolve())
    for evaluation_index, claim in enumerate(claims):
        loaded_model = LoadedEvaluationModel(
            identity=claim.candidate,
            model=load_evaluation_model(
                game,
                architecture,
                model_artifacts.load(claim.candidate),
                evaluation_device,
            ),
        )
        evaluation_id = derive_evaluation_id(
            state.run_id,
            state.resolved_configuration_sha256,
            search_sha256,
            evaluation_index,
            claim.requested_elapsed_seconds,
            claim.candidate,
            opponent,
            game,
        )
        specification = PairedEvaluationSpecification(
            evaluation_id=evaluation_id,
            run_id=state.run_id,
            resolved_configuration_sha256=state.resolved_configuration_sha256,
            common_search_sha256=search_sha256,
            evaluation_index=evaluation_index,
            root_seed=configuration.experiment.root_seed,
            requested_elapsed_seconds=claim.requested_elapsed_seconds,
            published_checkpoint_elapsed_seconds=claim.published_elapsed_seconds,
            candidate=claim.candidate,
            opponent=opponent,
            game=game,
        )
        with ExitStack() as stack:
            broker = stack.enter_context(
                evaluation_broker(
                    loaded_model.model,
                    game,
                    configuration,
                    evaluation_device,
                )
            )
            opponent_player = (
                RandomGoEvaluationPlayer()
                if opponent_model is None
                else NativeCheckpointEvaluationPlayer(
                    stack.enter_context(
                        evaluation_broker(
                            opponent_model.model,
                            game,
                            configuration,
                            evaluation_device,
                        )
                    ),
                    configuration.evaluation.search,
                    opponent_model,
                )
            )
            evaluator = PairedGoEvaluator(
                specification,
                NativeCheckpointEvaluationPlayer(
                    broker,
                    configuration.evaluation.search,
                    loaded_model,
                ),
                opponent_player,
                result_repository,
            )
            for pair_index in range(configuration.evaluation.paired_games_per_checkpoint // 2):
                if repository.stop_requested():
                    return interrupt_phase(
                        repository,
                        state,
                        state.self_play_elapsed_seconds,
                        (),
                    )
                evaluator.evaluate_pair(pair_index)
    expected_paths = tuple(
        result_repository.path(
            derive_evaluation_id(
                state.run_id,
                state.resolved_configuration_sha256,
                search_sha256,
                evaluation_index,
                claim.requested_elapsed_seconds,
                claim.candidate,
                opponent,
                game,
            ),
            pair_index,
            game_in_pair,
        )
        for evaluation_index, claim in enumerate(claims)
        for pair_index in range(configuration.evaluation.paired_games_per_checkpoint // 2)
        for game_in_pair in (0, 1)
    )
    require_exact_artifact_files(
        repository.directory / 'evaluation-results',
        '*.json',
        expected_paths,
    )
    artifacts = tuple(repository.artifact(RunArtifactKind.EVALUATION_RESULT, path) for path in expected_paths)
    return complete_phase(
        repository,
        state,
        ExperimentPhase.EVALUATION,
        artifacts,
    )


def load_checkpoint_claims(
    repository: ExperimentRunRepository,
    state: ExperimentRunState,
) -> tuple[ScheduledCheckpointClaim, ...]:
    paths = registered_artifact_paths(
        repository,
        state,
        RunArtifactKind.CHECKPOINT_CLAIM,
    )
    require_exact_artifact_files(
        repository.directory / 'checkpoint-claims',
        '*.json',
        paths,
    )
    claims = tuple(ScheduledCheckpointClaim.model_validate_json(path.read_bytes()) for path in paths)
    if not claims:
        raise ValueError('Evaluation requires at least one elapsed-time checkpoint claim.')
    if any(
        claim.run_id != state.run_id or claim.resolved_configuration_sha256 != state.resolved_configuration_sha256
        for claim in claims
    ):
        raise ValueError('Checkpoint claim does not belong to the active run identity.')
    requested = tuple(claim.requested_elapsed_seconds for claim in claims)
    if tuple(sorted(set(requested))) != requested:
        raise ValueError('Checkpoint claims must have unique increasing requested times.')
    return claims


def evaluation_pair(
    evaluation_id: UUID,
    pair_index: int,
    games: tuple[EvaluationGameResult, ...],
) -> EvaluationPairResult:
    selected = tuple(
        sorted(
            (game for game in games if game.pair_index == pair_index),
            key=lambda game: game.game_in_pair,
        )
    )
    if (
        len(selected) != 2
        or tuple(game.game_in_pair for game in selected) != (0, 1)
        or any(game.evaluation_id != evaluation_id for game in selected)
    ):
        raise ValueError('An evaluation pair requires exactly games zero and one.')
    return EvaluationPairResult(
        evaluation_id=evaluation_id,
        pair_index=pair_index,
        games=(selected[0], selected[1]),
    )


def load_evaluation_opponent(
    configuration: ResolvedRunConfiguration,
    game: GoGameConfiguration,
    architecture: ResidualGoModelConfiguration,
    device: torch.device,
    run_directory: Path,
) -> tuple[EvaluationOpponentIdentity, LoadedEvaluationModel | None]:
    match configuration.evaluation.suite.opponent:
        case RandomGoOpponent():
            return RandomOpponentIdentity(kind='random'), None
        case CheckpointGoOpponent(checkpoint=reference):
            artifact_root = (run_directory / 'reference-artifacts').resolve()
            manifest_path = artifact_root.joinpath(*reference.manifest_path.parts).resolve()
            model_path = artifact_root.joinpath(*reference.model_path.parts).resolve()
            if (
                artifact_root not in manifest_path.parents
                or artifact_root not in model_path.parents
                or not manifest_path.is_file()
                or not model_path.is_file()
            ):
                raise ValueError('Evaluation checkpoint opponent is outside its authenticated artifact root.')
            manifest_contents = manifest_path.read_bytes()
            if hashlib.sha256(manifest_contents).hexdigest() != reference.manifest_sha256:
                raise ValueError('Evaluation checkpoint opponent manifest checksum mismatch.')
            manifest = ModelCheckpointManifest.model_validate_json(manifest_contents)
            artifact = model_path.read_bytes()
            if (
                hashlib.sha256(artifact).hexdigest() != reference.model_artifact_sha256
                or manifest.model.sha256 != reference.model_artifact_sha256
                or manifest.model.filename != model_path.name
            ):
                raise ValueError('Evaluation checkpoint opponent model does not match its manifest.')
            identity = CandidateCheckpointIdentity(
                checkpoint_id=manifest.checkpoint_id,
                model_artifact_sha256=manifest.model.sha256,
                model_version=manifest.model_version,
            )
            loaded = LoadedEvaluationModel(
                identity=identity,
                model=load_evaluation_model(
                    game,
                    architecture,
                    artifact,
                    device,
                ),
            )
            return CheckpointOpponentIdentity(
                kind='checkpoint',
                checkpoint=identity,
            ), loaded


def load_evaluation_model(
    game: GoGameConfiguration,
    architecture: ResidualGoModelConfiguration,
    artifact: bytes,
    device: torch.device,
) -> ResidualGoModel:
    model = ResidualGoModel(game, architecture).to(device)
    model.load_state_dict(torch.load(io.BytesIO(artifact), map_location=device, weights_only=True))
    return model


def evaluation_broker(
    model: ResidualGoModel,
    game: GoGameConfiguration,
    configuration: ResolvedRunConfiguration,
    device: torch.device,
) -> GoInferenceBatchBroker:
    return GoInferenceBatchBroker(
        model=model,
        configuration=game,
        device=device,
        maximum_batch_size=configuration.evaluation.search.inference.maximum_batch_size,
        maximum_wait_microseconds=configuration.evaluation.search.inference.maximum_wait_microseconds,
        maximum_pending_batches=1,
        cache_capacity=configuration.evaluation.search.inference.cache_capacity,
    )
