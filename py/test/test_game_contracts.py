from pathlib import Path
from uuid import UUID

import pytest
import torch
from AlphaZeroCpp import GameSearchVisit
from pydantic import ValidationError

from src.experiment.configuration import load_experiment_configuration
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.chess.training import ChessImplementation
from src.games.go.contract import GoStateContract
from src.games.contracts import Player, WdlTarget
from src.replay.contracts import (
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    IneligibleNextPolicyTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    TerminationReason,
    publish_completed_self_play_game,
)
from src.training.objective import (
    ResolvedNextPolicyLoss,
    ResolvedRemainingGameLengthLoss,
    ResolvedTrainingObjective,
)
from src.training.targets import RemainingGameLengthHeadLayout, build_training_target_layout
from src.training.batch import TrainingBatch, TrainingModelOutput


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
    ((Path('configs/chess-experiment-template.yaml'), 1_880, 2),),
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
    assert implementation.target_layout.auxiliary_heads == ()


def test_completed_self_play_game_round_trip_uses_shared_trajectory_values() -> None:
    observation = SearchObservation(
        ply=0,
        model_generation=3,
        policy_target_visits=(GameSearchVisit(action_id=7, visit_count=12),),
        root_value=0.25,
        highest_visited_child_action_id=7,
        highest_visited_child_visit_count=12,
        highest_visited_child_q=0.2,
        selected_action_id=7,
        full_search=True,
        sample_weight=1.0,
        search_budget=16,
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
    configuration = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    assert configuration.game == 'chess'
    objective = configuration.chess.objective.validated_copy(
        update={
            'auxiliary_targets': [
                {
                    'kind': 'next_policy',
                    'ply_offset': 1,
                    'loss_weight': {'kind': 'constant', 'value': 0.25},
                },
                {
                    'kind': 'next_policy',
                    'ply_offset': 3,
                    'loss_weight': {'kind': 'constant', 'value': 0.1},
                },
            ]
        }
    )
    layout = build_training_target_layout(CHESS_STATE_CONTRACT.action_size, objective.auxiliary_targets)

    assert tuple(head.ply_offset for head in layout.auxiliary_heads) == (1, 3)
    assert all(head.action_size == 1_880 for head in layout.auxiliary_heads)

    chess_configuration = configuration.chess.validated_copy(update={'objective': objective.model_dump(mode='json')})
    resolved = configuration.validated_copy(update={'chess': chess_configuration.model_dump(mode='json')})
    assert resolved.chess.objective.auxiliary_targets == objective.auxiliary_targets


def test_next_policy_eligibility_is_explicit_and_uses_future_action_space() -> None:
    future_policy = SparsePolicyTarget(visits=(GameSearchVisit(action_id=42, visit_count=10),))

    eligible = EligibleNextPolicyTarget(policy=future_policy)
    terminal_tail = IneligibleNextPolicyTarget()

    assert eligible.eligible
    assert eligible.policy.visits[0].action_id == 42
    assert not terminal_tail.eligible


def test_remaining_game_length_layout_is_a_scalar_target() -> None:
    configuration = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    assert configuration.game == 'chess'
    objective = configuration.chess.objective.validated_copy(
        update={
            'auxiliary_targets': [
                {
                    'kind': 'remaining_game_length',
                    'normalization_scale': 256,
                    'loss_weight': {'kind': 'constant', 'value': 0.15},
                }
            ]
        }
    )

    layout = build_training_target_layout(CHESS_STATE_CONTRACT.action_size, objective.auxiliary_targets)

    assert layout.auxiliary_heads == (
        RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=256),
    )


def test_search_observation_rejects_zero_visit_entries() -> None:
    with pytest.raises(ValueError, match='positive'):
        GameSearchVisit(action_id=1, visit_count=0)


def test_native_search_visit_has_readable_value_semantics() -> None:
    visit = GameSearchVisit(action_id=7, visit_count=12)

    assert visit == GameSearchVisit(action_id=7, visit_count=12)
    assert repr(visit) == 'GameSearchVisit(action_id=7, visit_count=12)'


def test_chess_augmentation_transforms_state_primary_and_auxiliary_policy_together() -> None:
    pytest.importorskip('AlphaZeroCpp')
    position = CHESS_STATE_CONTRACT.initial_position()
    action_id = CHESS_STATE_CONTRACT.legal_action_ids(position)[0]
    policy = SparsePolicyTarget(visits=(GameSearchVisit(action_id=action_id, visit_count=3),))
    sample = ReplaySample(
        encoded_state=CHESS_STATE_CONTRACT.encode_network_input(position),
        policy=policy,
        wdl_target=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        root_value=0.0,
        auxiliary_targets=(
            EligibleNextPolicyTarget(policy=policy),
            IneligibleNextPolicyTarget(),
            EligibleRemainingGameLengthTarget(normalized_length=0.5),
        ),
        sample_weight=1.0,
        source_model_generation=0,
        source_created_at_seconds=1.0,
    )

    transformed = CHESS_STATE_CONTRACT.transform_replay_targets(sample, augmentation_index=1)
    transformed_action = CHESS_STATE_CONTRACT.transform_action_id(action_id, augmentation_index=1)

    assert transformed.policy.visits[0].action_id == transformed_action
    auxiliary = transformed.auxiliary_targets[0]
    assert isinstance(auxiliary, EligibleNextPolicyTarget)
    assert auxiliary.policy.visits[0].action_id == transformed_action
    assert transformed.auxiliary_targets[1] == IneligibleNextPolicyTarget()
    assert transformed.auxiliary_targets[2] == EligibleRemainingGameLengthTarget(normalized_length=0.5)
    assert transformed.encoded_state == CHESS_STATE_CONTRACT.transform_encoded_state(sample.encoded_state, 1)


def test_canonical_batch_and_model_output_are_the_objective_boundary() -> None:
    configuration = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    assert configuration.game == 'chess'
    objective = ChessImplementation(configuration).training_objective_at(0)
    batch = TrainingBatch(
        states=torch.zeros((2, 1)),
        policy_targets=torch.tensor(((1.0, 0.0), (0.0, 1.0))),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        root_values=torch.tensor((0.5, -0.5)),
        auxiliary_targets=(),
        auxiliary_eligibility=(),
        sample_weights=torch.ones(2),
        source_model_generations=torch.zeros(2, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(2, dtype=torch.float64),
    )
    output = TrainingModelOutput(
        policy_logits=torch.tensor(((2.0, 0.0), (0.0, 2.0))),
        wdl_logits=torch.tensor(((2.0, 0.0, 0.0), (0.0, 2.0, 0.0))),
        auxiliary_logits=(),
    )

    loss = objective.calculate_loss(output, batch)

    assert loss.total.isfinite()
    assert loss.auxiliary == ()


def test_remaining_game_length_uses_masked_smooth_l1_loss() -> None:
    objective = ResolvedTrainingObjective(
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedRemainingGameLengthLoss(weight=0.15),),
    )
    batch = TrainingBatch(
        states=torch.zeros((2, 1)),
        policy_targets=torch.tensor(((1.0, 0.0), (1.0, 0.0))),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        root_values=torch.zeros(2),
        auxiliary_targets=(torch.tensor(((0.5,), (1.0,))),),
        auxiliary_eligibility=(torch.tensor((True, False)),),
        sample_weights=torch.ones(2),
        source_model_generations=torch.zeros(2, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(2, dtype=torch.float64),
    )
    output = TrainingModelOutput(
        policy_logits=torch.zeros((2, 2)),
        wdl_logits=torch.zeros((2, 3)),
        auxiliary_logits=(torch.tensor(((0.0,), (100.0,))),),
    )

    loss = objective.calculate_loss(output, batch)

    assert loss.auxiliary[0].item() == pytest.approx(0.125)
    assert loss.total.item() == pytest.approx(0.15 * 0.125)


def test_next_policy_auxiliary_still_uses_masked_cross_entropy() -> None:
    objective = ResolvedTrainingObjective(
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedNextPolicyLoss(weight=0.15),),
    )
    batch = TrainingBatch(
        states=torch.zeros((2, 1)),
        policy_targets=torch.tensor(((1.0, 0.0), (1.0, 0.0))),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        root_values=torch.zeros(2),
        auxiliary_targets=(torch.tensor(((1.0, 0.0), (0.0, 1.0))),),
        auxiliary_eligibility=(torch.tensor((True, False)),),
        sample_weights=torch.ones(2),
        source_model_generations=torch.zeros(2, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(2, dtype=torch.float64),
    )
    output = TrainingModelOutput(
        policy_logits=torch.zeros((2, 2)),
        wdl_logits=torch.zeros((2, 3)),
        auxiliary_logits=(torch.tensor(((2.0, 0.0), (100.0, -100.0))),),
    )

    loss = objective.calculate_loss(output, batch)

    assert loss.auxiliary[0].item() == pytest.approx(torch.log1p(torch.exp(torch.tensor(-2.0))).item())
