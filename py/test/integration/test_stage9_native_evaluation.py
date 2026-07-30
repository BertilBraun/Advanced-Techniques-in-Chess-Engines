from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
import torch

native = pytest.importorskip('az_go_native', reason='focused native Go extension has not been built')

from src.az.calibration.calibrate import committed_trace_observations
from src.az.calibration.models import (
    InitialTraceModelIdentity,
    load_trace_collection_artifact,
    publish_trace_collection_artifact,
)
from src.az.config.search import (
    ConstantTemperature,
    DisabledRootExploration,
    DisabledTreeReuse,
    FixedSearchBudget,
    FullBudgetStopping,
    PuctSearchConfiguration,
    SearchConfiguration,
    SearchInferenceConfiguration,
    VisitedChildMeanFpu,
)
from src.az.config.serialization import model_sha256
from src.az.evaluation.models import CandidateCheckpointIdentity, RandomOpponentIdentity
from src.az.evaluation.protocol import (
    NativeCheckpointEvaluationPlayer,
    LoadedEvaluationModel,
    PairedEvaluationSpecification,
    PairedGoEvaluator,
    RandomGoEvaluationPlayer,
    derive_evaluation_id,
)
from src.az.evaluation.storage import EvaluationResultRepository
from src.az.games.go.configuration import DisabledResignation, GoGameConfiguration, ResidualGoModelConfiguration
from src.az.games.go.model import ResidualGoModel
from src.az.inference.go_batching import GoInferenceBatchBroker
from src.az.self_play.configuration import GoWorkerSpecification, NativeSearchSpecification
from src.az.self_play.worker import _sample_trace, _search_trace_payload


def test_real_native_checkpoint_search_pairs_against_random_with_balanced_colors(tmp_path: Path) -> None:
    game = GoGameConfiguration(
        kind='go',
        board_size=3,
        komi_half_points=1,
        scoring_rule='area',
        ko_rule='positional_superko',
        suicide_rule='illegal',
        pass_exempt_from_superko=True,
        score_comparison='doubled_integer_points',
        safety_ply_cap=18,
        history_length=2,
        history_planes_per_position=2,
        include_color_plane=True,
        pass_action='last',
        normal_termination='two_consecutive_passes',
        symmetry_group='dihedral_8',
        capped_game_value_target_weight=0,
        resignation=DisabledResignation(kind='disabled'),
    )
    model_configuration = ResidualGoModelConfiguration(
        family='residual_go',
        channels=4,
        residual_blocks=1,
        policy_channels=2,
        value_hidden_size=4,
        normalization='batch',
        activation='relu',
    )
    search = SearchConfiguration(
        algorithm=PuctSearchConfiguration(kind='puct', exploration_constant=1.5),
        budget=FixedSearchBudget(kind='fixed', simulations=2),
        stopping=FullBudgetStopping(kind='full_budget'),
        fpu=VisitedChildMeanFpu(kind='visited_child_mean', no_visited_child_value=0),
        root_exploration=DisabledRootExploration(kind='disabled'),
        temperature=ConstantTemperature(kind='constant', temperature=0),
        tree_reuse=DisabledTreeReuse(kind='disabled'),
        inference=SearchInferenceConfiguration(maximum_batch_size=2, maximum_wait_microseconds=0, cache_capacity=0),
        backup_discount=1,
    )
    torch.manual_seed(7)
    model = ResidualGoModel(game, model_configuration)
    run_id = UUID(int=699)
    candidate = CandidateCheckpointIdentity(
        checkpoint_id=UUID(int=701),
        model_artifact_sha256='a' * 64,
        model_version=1,
    )
    opponent = RandomOpponentIdentity(kind='random')
    configuration_sha256 = 'b' * 64
    search_sha256 = model_sha256(search)
    evaluation_id = derive_evaluation_id(
        run_id,
        configuration_sha256,
        search_sha256,
        0,
        10,
        candidate,
        opponent,
        game,
    )
    with GoInferenceBatchBroker(
        model=model,
        configuration=game,
        device=torch.device('cpu'),
        maximum_batch_size=2,
        maximum_wait_microseconds=0,
        maximum_pending_batches=2,
        cache_capacity=0,
    ) as broker:
        specification = PairedEvaluationSpecification(
            evaluation_id=evaluation_id,
            run_id=run_id,
            resolved_configuration_sha256=configuration_sha256,
            common_search_sha256=search_sha256,
            evaluation_index=0,
            root_seed=71,
            requested_elapsed_seconds=10,
            published_checkpoint_elapsed_seconds=12.5,
            candidate=candidate,
            opponent=opponent,
            game=game,
        )
        with pytest.raises(ValueError, match='model does not match'):
            PairedGoEvaluator(
                specification,
                NativeCheckpointEvaluationPlayer(
                    broker,
                    search,
                    LoadedEvaluationModel(
                        identity=candidate.model_copy(update={'checkpoint_id': UUID(int=999)}),
                        model=model,
                    ),
                ),
                RandomGoEvaluationPlayer(),
                EvaluationResultRepository(tmp_path.resolve()),
            )
        evaluator = PairedGoEvaluator(
            specification,
            NativeCheckpointEvaluationPlayer(
                broker,
                search,
                LoadedEvaluationModel(identity=candidate, model=model),
            ),
            RandomGoEvaluationPlayer(),
            EvaluationResultRepository(tmp_path.resolve()),
        )
        pair = evaluator.evaluate_pair(0)

        trace_directory = (tmp_path / 'traces').resolve()
        worker = GoWorkerSpecification(
            worker_index=0,
            process_index=0,
            run_id=run_id,
            root_seed=71,
            game_configuration=game,
            model_configuration=model_configuration,
            model_initialization_seed=7,
            search=NativeSearchSpecification(
                budget=search.budget,
                stopping=search.stopping,
                fpu=search.fpu,
                exploration_constant=search.algorithm.exploration_constant,
                backup_discount=search.backup_discount,
                temperature=search.temperature,
                root_exploration=search.root_exploration,
            ),
            logical_worker_start_index=0,
            logical_worker_count=1,
            next_game_indices=(0,),
            maximum_active_searches_per_worker=1,
            maximum_batch_size=2,
            maximum_wait_microseconds=0,
            maximum_pending_batches=2,
            inference_cache_capacity=0,
            value_target_weight=1,
            device='cpu',
            torch_intraop_thread_count=1,
            checkpoint_directory=str(tmp_path.resolve()),
            resolved_configuration_sha256=configuration_sha256,
            telemetry_write_every_seconds=1,
            resource_sample_every_seconds=1,
            search_trace_sample_probability=1,
            search_trace_checkpoints=(1,),
            search_trace_directory=str(trace_directory),
        )
        assert _sample_trace(worker, 0, 0, 0)
        state = native.GoState(native.GoRules(3, 1, 18, 2))
        encoding = state.canonical_encoding()
        legal_actions = tuple(state.legal_actions())
        traced = native.search_go_fixed(
            state,
            broker.evaluate,
            native.FixedPuctConfiguration(
                2,
                1.5,
                1.0,
                0.0,
                0.0,
                1,
                2,
                native.RootNoiseConfiguration(False, 1.0, 0.0),
                False,
                native.FpuPolicy.VISITED_CHILD_MEAN,
                0.0,
                native.AdaptiveStoppingConfiguration(False, 1, 1, 1.0, 1.0),
                native.SearchBudgetClass.FIXED,
                0.0,
                native.PrefixTraceConfiguration(True, [1]),
            ),
        )
        payload = _search_trace_payload(
            worker,
            0,
            0,
            InitialTraceModelIdentity(
                kind='initial_model',
                model_initialization_seed=7,
                model_configuration_sha256=model_sha256(model_configuration),
            ),
            UUID(int=800),
            0,
            encoding,
            legal_actions,
            state.state_hash(),
            traced,
        )
        trace_path = publish_trace_collection_artifact(trace_directory, payload)
        loaded_trace = load_trace_collection_artifact(trace_path)
        assert committed_trace_observations(
            (loaded_trace.artifact,),
            frozenset((payload.replay_sample_id,)),
        ) == (payload.observation,)
        zero_sampling = worker.model_copy(update={'search_trace_sample_probability': 0})
        before = tuple(trace_directory.iterdir())
        assert not _sample_trace(zero_sampling, 0, 0, 0)
        assert tuple(trace_directory.iterdir()) == before

    assert tuple(game_result.candidate_color.value for game_result in pair.games) == ('black', 'white')
    assert all(game_result.komi_half_points == 1 for game_result in pair.games)
    assert all(game_result.candidate_actual_simulations > 0 for game_result in pair.games)
    assert all(game_result.opponent_actual_simulations == 0 for game_result in pair.games)
    assert evaluator.evaluate_pair(0) == pair
