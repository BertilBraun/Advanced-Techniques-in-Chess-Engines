from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
import torch

pytest.importorskip('AlphaZeroCpp')
from AlphaZeroCpp import GameSearchVisit
from pydantic import ValidationError
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.chess.training import ChessImplementation
from src.games.contracts import Player, WdlTarget
from src.games.go.contract import GoStateContract
from src.replay.contracts import (
    EligibleNextPolicyTarget,
    IneligibleNextPolicyTarget,
    SparsePolicyTarget,
)
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
    TerminationReason,
    publish_completed_self_play_game,
)
from src.training.batch import TrainingModelOutput
from src.training.objective import (
    ResolvedLegalMovesLoss,
    ResolvedNextPolicyLoss,
    ResolvedRemainingGameLengthLoss,
    ResolvedTrainingObjective,
)
from src.training.targets import (
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchBudgetHeadLayout,
    build_training_target_layout,
)
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY
from test_helpers.training_batches import training_batch


def test_wdl_target_validates_and_reverses_perspective() -> None:
    target = WdlTarget.from_scalar(0.4)

    assert (target.win, target.draw, target.loss) == pytest.approx((0.6, 0.2, 0.2))
    reversed_target = target.reversed()
    assert (reversed_target.win, reversed_target.draw, reversed_target.loss) == pytest.approx((0.2, 0.2, 0.6))
    with pytest.raises(ValidationError, match='sum to one'):
        WdlTarget(win=0.5, draw=0.5, loss=0.5)


def test_chess_state_contract_operates_in_action_id_space() -> None:
    pytest.importorskip('AlphaZeroCpp')
    initial = CHESS_STATE_CONTRACT.initial_position()
    legal_actions = CHESS_STATE_CONTRACT.legal_action_ids(initial)
    child = CHESS_STATE_CONTRACT.child_position(initial, legal_actions[0])

    assert legal_actions
    assert all(initial.action_id_from_uci(initial.action_uci(action_id)) == action_id for action_id in legal_actions)
    assert initial.fen.startswith('rnbqkbnr/pppppppp/')
    assert child.fen != initial.fen
    assert CHESS_STATE_CONTRACT.current_player(initial) is Player.FIRST
    assert CHESS_STATE_CONTRACT.current_player(child) is Player.SECOND
    assert CHESS_STATE_CONTRACT.natural_terminal_wdl(initial) is None
    assert (
        len(CHESS_STATE_CONTRACT.encode_network_input(initial))
        == CHESS_STATE_CONTRACT.packed_plane_layout.payload_bytes
    )


def test_chess_adjudication_uses_fixed_starting_material_normalization() -> None:
    native = pytest.importorskip('AlphaZeroCpp')
    position = native.ChessPosition('4k3/8/8/8/8/8/8/3QK3 w - - 0 1')

    target = CHESS_STATE_CONTRACT.adjudicated_wdl(position, TerminationReason.MAXIMUM_PLIES)

    assert target == WdlTarget.from_scalar(9 / 39)


@pytest.mark.parametrize(
    ('komi_half_points', 'expected'),
    (
        (0, WdlTarget(win=0.0, draw=1.0, loss=0.0)),
        (15, WdlTarget(win=0.0, draw=0.0, loss=1.0)),
    ),
)
def test_go_adjudication_area_scores_nonterminal_positions(
    komi_half_points: int,
    expected: WdlTarget,
) -> None:
    pytest.importorskip('AlphaZeroCpp')
    state = GoStateContract(board_size=7, komi_half_points=komi_half_points, maximum_moves=98)

    target = state.adjudicated_wdl(state.initial_position(), TerminationReason.MAXIMUM_PLIES)

    assert target == expected


@pytest.mark.parametrize(
    ('path', 'expected_action_size', 'expected_augmentations'),
    ((TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml', 1_880, 2),),
)
def test_root_game_implementation_owns_state_and_fixed_target_layout(
    path: Path,
    expected_action_size: int,
    expected_augmentations: int,
) -> None:
    configuration = load_experiment_configuration(path)
    assert configuration.game == 'chess'
    implementation = ChessImplementation(configuration)

    assert implementation.state.action_size == expected_action_size
    assert implementation.state.augmentation_count == expected_augmentations
    assert implementation.target_layout.action_size == expected_action_size
    assert implementation.target_layout.wdl_size == 3
    assert implementation.target_layout.auxiliary_heads == (SearchBudgetHeadLayout(kind='search_budget'),)


def test_completed_self_play_game_round_trip_uses_shared_trajectory_values() -> None:
    observation = SearchObservation(
        ply=0,
        model_generation=3,
        policy_target_visits=SearchVisitCounts.from_native((GameSearchVisit(action_id=7, visit_count=12),)),
        root_value=0.25,
        highest_visited_child_action_id=7,
        highest_visited_child_visit_count=12,
        highest_visited_child_q=0.2,
        selected_action_id=7,
        sample_weight=1.0,
        baseline_visits=16,
        network_root_value=0.1,
        policy_correction=0.2,
        value_correction=0.075,
        search_budget_logit=-0.4,
        predicted_search_budget=0.4,
        assigned_additional_visits=16,
        parallel_searches=1,
        spend_residual=0,
        starting_visits=0,
        final_visits=16,
        stop_reason=SearchStopReason.FIXED_LIMIT,
    )
    game = CompletedSelfPlayGame(
        identity=GameIdentity(
            worker_id=2,
            process_instance_id=UUID('12345678-1234-5678-1234-567812345678'),
            game_number=3,
        ),
        created_at_seconds=4.0,
        generation_seconds=5.0,
        action_ids=(7,),
        observations=(observation,),
        final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        termination_reason=TerminationReason.NATURAL,
    )

    assert CompletedSelfPlayGame.model_validate_json(game.model_dump_json()) == game
    with pytest.raises(ValidationError, match='selected actions'):
        game.validated_copy(update={'action_ids': [8]})


def test_canonical_completed_game_publication_needs_no_publisher_state(tmp_path: Path) -> None:
    game = CompletedSelfPlayGame(
        identity=GameIdentity(
            worker_id=4,
            process_instance_id=UUID('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'),
            game_number=12,
        ),
        created_at_seconds=1.0,
        generation_seconds=2.0,
        action_ids=(),
        observations=(),
        final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        termination_reason=TerminationReason.NATURAL,
    )

    path = publish_completed_self_play_game(tmp_path, game)

    assert path.name == 'worker-4-process-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee-game-00000000000000000012.json'
    assert CompletedSelfPlayGame.model_validate_json(path.read_text(encoding='utf-8')) == game


def test_auxiliary_target_layout_is_run_fixed_and_ordered() -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert configuration.game == 'chess'
    objective = configuration.chess.objective.validated_copy(
        update={
            'auxiliary_targets': [
                {
                    'kind': 'next_policy',
                    'ply_offset': 1,
                    'loss_weight': 0.25,
                },
                {
                    'kind': 'next_policy',
                    'ply_offset': 3,
                    'loss_weight': 0.1,
                },
                {
                    'kind': 'search_budget',
                    'loss_weight': 0.2,
                },
            ]
        }
    )
    layout = build_training_target_layout(CHESS_STATE_CONTRACT.action_size, objective.auxiliary_targets)

    assert layout.auxiliary_heads == (
        NextPolicyHeadLayout(kind='next_policy', action_size=1_880, ply_offset=1),
        NextPolicyHeadLayout(kind='next_policy', action_size=1_880, ply_offset=3),
        SearchBudgetHeadLayout(kind='search_budget'),
    )

    chess_configuration = configuration.chess.validated_copy(update={'objective': objective.model_dump(mode='json')})
    resolved = configuration.validated_copy(update={'chess': chess_configuration.model_dump(mode='json')})
    assert resolved.chess.objective.auxiliary_targets == objective.auxiliary_targets


def test_next_policy_eligibility_is_explicit_and_uses_future_action_space() -> None:
    future_policy = SparsePolicyTarget(
        visits=SearchVisitCounts.from_native((GameSearchVisit(action_id=42, visit_count=10),)),
        legal_action_ids=(42, 43),
    )

    eligible = EligibleNextPolicyTarget(policy=future_policy)
    terminal_tail = IneligibleNextPolicyTarget()

    assert eligible.eligible
    assert eligible.policy.visits.action_ids[0] == 42
    assert not terminal_tail.eligible


def test_remaining_game_length_layout_is_a_scalar_target() -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert configuration.game == 'chess'
    objective = configuration.chess.objective.validated_copy(
        update={
            'auxiliary_targets': [
                {
                    'kind': 'remaining_game_length',
                    'normalization_scale': 256,
                    'loss_weight': 0.15,
                }
            ]
        }
    )

    layout = build_training_target_layout(CHESS_STATE_CONTRACT.action_size, objective.auxiliary_targets)

    assert layout.auxiliary_heads == (
        RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=256),
    )


def test_native_search_visit_rejects_zero_visit_count() -> None:
    with pytest.raises(ValueError, match='positive'):
        GameSearchVisit(action_id=1, visit_count=0)


def test_native_search_visit_has_readable_value_semantics() -> None:
    visit = GameSearchVisit(action_id=7, visit_count=12)

    assert visit == GameSearchVisit(action_id=7, visit_count=12)
    assert repr(visit) == 'GameSearchVisit(action_id=7, visit_count=12)'


def test_chess_action_permutations_are_cached_read_only_bijections() -> None:
    pytest.importorskip('AlphaZeroCpp')
    position = CHESS_STATE_CONTRACT.initial_position()
    action_id = CHESS_STATE_CONTRACT.legal_action_ids(position)[0]
    permutations = CHESS_STATE_CONTRACT.action_permutations

    assert permutations is CHESS_STATE_CONTRACT.action_permutations
    assert not permutations.flags.writeable
    assert permutations[1, action_id] == CHESS_STATE_CONTRACT.transform_action_id(action_id, 1)


def test_canonical_batch_and_model_output_are_the_objective_boundary() -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert configuration.game == 'chess'
    objective = ChessImplementation(configuration).training_objective_at(0)
    batch = training_batch(
        policy_targets=torch.tensor(((1.0, 0.0), (0.0, 1.0))),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        root_values=torch.tensor((0.5, -0.5)),
        auxiliary_targets=(torch.zeros((2, 1)),),
        auxiliary_legal_action_ids=(torch.empty((2, 0), dtype=torch.int64),),
        auxiliary_eligibility=(torch.tensor((False, False)),),
    )
    output = TrainingModelOutput(
        policy_logits=torch.tensor(((2.0, 0.0), (0.0, 2.0))),
        wdl_logits=torch.tensor(((2.0, 0.0, 0.0), (0.0, 2.0, 0.0))),
        auxiliary_logits=(torch.zeros((2, 1)),),
        features=torch.empty((2, 0)),
    )

    loss = objective.calculate_loss(output, batch)

    assert loss.total.isfinite()
    assert loss.auxiliary[0].item() == 0.0


def test_policy_loss_normalizes_only_over_legal_actions() -> None:
    objective = ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=0.0,
        root_value_blend=0.0,
        auxiliary_losses=(),
    )
    batch = training_batch(
        policy_targets=torch.tensor(((1.0, 0.0, 0.0),)),
        wdl_targets=torch.tensor(((0.0, 1.0, 0.0),)),
    )
    output = TrainingModelOutput(
        policy_logits=torch.tensor(((0.0, 0.0, 100.0),)),
        wdl_logits=torch.zeros((1, 3)),
        auxiliary_logits=(),
        features=torch.empty((1, 0)),
    )

    loss = objective.calculate_loss(output, batch)

    assert loss.policy.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_remaining_game_length_uses_masked_smooth_l1_loss() -> None:
    objective = ResolvedTrainingObjective(
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedRemainingGameLengthLoss(weight=0.15),),
    )
    batch = training_batch(
        policy_targets=torch.tensor(((1.0, 0.0), (1.0, 0.0))),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        auxiliary_targets=(torch.tensor(((0.5,), (1.0,))),),
        auxiliary_legal_action_ids=(torch.empty((2, 0), dtype=torch.int64),),
        auxiliary_eligibility=(torch.tensor((True, False)),),
    )
    output = TrainingModelOutput(
        policy_logits=torch.zeros((2, 2)),
        wdl_logits=torch.zeros((2, 3)),
        auxiliary_logits=(torch.tensor(((0.0,), (100.0,))),),
        features=torch.empty((2, 0)),
    )

    loss = objective.calculate_loss(output, batch)

    assert loss.auxiliary[0].item() == pytest.approx(0.125)
    assert loss.total.item() == pytest.approx(0.15 * 0.125)


@pytest.mark.parametrize('prediction_dtype', (torch.float32, torch.bfloat16))
def test_next_policy_auxiliary_still_uses_masked_cross_entropy(prediction_dtype: torch.dtype) -> None:
    objective = ResolvedTrainingObjective(
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedNextPolicyLoss(weight=0.15),),
    )
    batch = training_batch(
        policy_targets=torch.tensor(((1.0, 0.0), (1.0, 0.0))),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        auxiliary_targets=(torch.tensor(((1.0, 0.0), (0.0, 1.0))),),
        auxiliary_legal_action_ids=(torch.tensor(((0, 1), (0, 1))),),
        auxiliary_eligibility=(torch.tensor((True, False)),),
    )
    output = TrainingModelOutput(
        policy_logits=torch.zeros((2, 2)),
        wdl_logits=torch.zeros((2, 3)),
        auxiliary_logits=(torch.tensor(((2.0, 0.0), (100.0, -100.0)), dtype=prediction_dtype),),
        features=torch.empty((2, 0)),
    )

    loss = objective.calculate_loss(output, batch)

    assert loss.auxiliary[0].item() == pytest.approx(
        torch.log1p(torch.exp(torch.tensor(-2.0))).item(),
        rel=0.02,
    )


def test_legal_move_loss_balances_legal_and_illegal_classes_per_position() -> None:
    objective = ResolvedTrainingObjective(
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedLegalMovesLoss(weight=1.0),),
    )
    target = torch.tensor(((1.0, 0.0, 0.0, 0.0),))
    prediction = torch.tensor(((0.0, 0.0, 2.0, -2.0),))
    batch = training_batch(
        policy_targets=torch.tensor(((1.0, 0.0),)),
        wdl_targets=torch.tensor(((0.0, 1.0, 0.0),)),
        auxiliary_targets=(target,),
        auxiliary_legal_action_ids=(torch.empty((1, 0), dtype=torch.int64),),
        auxiliary_eligibility=(torch.tensor((True,)),),
    )
    output = TrainingModelOutput(
        policy_logits=torch.zeros((1, 2)),
        wdl_logits=torch.zeros((1, 3)),
        auxiliary_logits=(prediction,),
        features=torch.empty((1, 0)),
    )
    element_loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction='none')
    expected = 0.5 * (element_loss[0, :1].mean() + element_loss[0, 1:].mean())

    loss = objective.calculate_loss(output, batch)

    assert loss.auxiliary[0].item() == pytest.approx(expected.item())
