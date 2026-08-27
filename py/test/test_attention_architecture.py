from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.games.chess.policy_encoding import (
    BOARD_SQUARE_COUNT,
    CHESS_FROM_TO_ACTION_TABLE,
    KNIGHT_PROMOTION_INDEX,
    PROMOTION_PIECES,
)
from src.training.checkpoint.persistence import create_model
from src.training.model_cost import measure_model_cost
from src.training.network import (
    BOOTSTRAP_POLICY_PRIOR_TARGET_TOP3_MASS,
    AttentionNetworkParams,
    ChessFromToAttentionPolicyHead,
    ChessFromToAttentionPolicyHeadConfiguration,
    DensePolicyHeadConfiguration,
    DisabledAttentionBiasConfiguration,
    GlobalPoolingResidualContext,
    InferenceNetwork,
    Network,
    NetworkParams,
    RelativeAttentionBiasConfiguration,
    ResidualContextPlacement,
    SmolgenAttentionBias,
    SmolgenAttentionBiasConfiguration,
    calibrate_bootstrap_policy_prior,
    measure_policy_prior_shape,
)
from test.test_distillation import STUDENT_ARGUMENTS
from tools.distill_train_student import (
    ALPHAZERO_SGD_STAGES,
    AttentionBiasKind,
    LearningRateSchedule,
    NetworkKind,
    OptimizerKind,
    PolicyHeadKind,
    create_student_optimizer,
    learning_rate_at,
    student_architecture,
)

DEVICE = torch.device('cpu')
SMOLGEN = SmolgenAttentionBiasConfiguration(compressed_size=8, hidden_size=32, generated_size=32)
FROM_TO_HEAD = ChessFromToAttentionPolicyHeadConfiguration(key_size=64)
DENSE_HEAD = DensePolicyHeadConfiguration(channels=4)


def attention_parameters(**overrides: object) -> AttentionNetworkParams:
    return AttentionNetworkParams(
        num_layers=overrides.pop('num_layers', 2),
        embedding_size=overrides.pop('embedding_size', 32),
        num_heads=overrides.pop('num_heads', 4),
        feedforward_size=overrides.pop('feedforward_size', 64),
        policy_head=overrides.pop('policy_head', FROM_TO_HEAD),
        **overrides,
    )


def chess_states(count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260827)
    return torch.rand((count, CHESS_NETWORK_DIMENSIONS.channels, 8, 8), generator=generator)


def fused_export(model: Network) -> InferenceNetwork:
    export = InferenceNetwork(model)
    export.eval()
    export.fuse_model()
    return export


def test_the_action_table_covers_the_whole_reduced_action_space() -> None:
    table = CHESS_FROM_TO_ACTION_TABLE

    assert table.action_count == CHESS_NETWORK_DIMENSIONS.actions
    assert len(table.promotion_action_ids) == 88
    assert set(table.from_squares.tolist()) == set(range(BOARD_SQUARE_COUNT))


def test_every_non_promotion_action_owns_a_distinct_square_pair() -> None:
    table = CHESS_FROM_TO_ACTION_TABLE
    plain = table.from_to_indices[table.promotion_indices < 0]

    assert len(set(plain.tolist())) == len(plain)


def test_every_promotion_action_shares_its_square_pair_with_a_plain_move() -> None:
    table = CHESS_FROM_TO_ACTION_TABLE
    plain = set(table.from_to_indices[table.promotion_indices < 0].tolist())
    promoting = set(table.from_to_indices[table.promotion_indices >= 0].tolist())

    assert promoting <= plain


@pytest.mark.native
def test_the_python_action_table_matches_the_native_chess_encoding() -> None:
    native = pytest.importorskip('AlphaZeroCpp')
    table = CHESS_FROM_TO_ACTION_TABLE
    position = native.ChessPosition()

    for action_id in position.legal_actions():
        uci = position.action_uci(action_id)
        from_square = (int(uci[1]) - 1) * 8 + (ord(uci[0]) - ord('a'))
        to_square = (int(uci[3]) - 1) * 8 + (ord(uci[2]) - ord('a'))
        promotion = PROMOTION_PIECES.index(uci[4]) if len(uci) > 4 else -1
        assert int(table.from_squares[action_id]) == from_square
        assert int(table.to_squares[action_id]) == to_square
        assert int(table.promotion_indices[action_id]) == promotion


@pytest.mark.native
def test_the_python_action_table_matches_the_native_encoding_after_a_promotion_opening() -> None:
    native = pytest.importorskip('AlphaZeroCpp')
    table = CHESS_FROM_TO_ACTION_TABLE
    position = native.ChessPosition('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
    promotions = 0

    for action_id in position.legal_actions():
        uci = position.action_uci(action_id)
        from_square = (int(uci[1]) - 1) * 8 + (ord(uci[0]) - ord('a'))
        to_square = (int(uci[3]) - 1) * 8 + (ord(uci[2]) - ord('a'))
        promotion = PROMOTION_PIECES.index(uci[4]) if len(uci) > 4 else -1
        promotions += promotion >= 0
        assert int(table.from_squares[action_id]) == from_square
        assert int(table.to_squares[action_id]) == to_square
        assert int(table.promotion_indices[action_id]) == promotion

    assert promotions == 4


def test_the_from_to_head_is_an_order_of_magnitude_smaller_than_the_dense_head() -> None:
    from_to = Network(attention_parameters(embedding_size=176, num_heads=11), DEVICE, CHESS_NETWORK_DIMENSIONS)
    dense = Network(
        attention_parameters(embedding_size=176, num_heads=11, policy_head=DENSE_HEAD),
        DEVICE,
        CHESS_NETWORK_DIMENSIONS,
    )

    from_to_head = sum(parameter.numel() for parameter in from_to.policy_head.parameters())
    dense_head = sum(parameter.numel() for parameter in dense.policy_head.parameters())

    assert from_to_head * 10 < dense_head


def test_the_from_to_head_scores_knight_promotion_at_the_plain_square_pair_logit() -> None:
    torch.manual_seed(13)
    head = ChessFromToAttentionPolicyHead(32, 16, 8, 8)
    table = CHESS_FROM_TO_ACTION_TABLE
    features = torch.rand((3, 32, 8, 8))

    logits = head(features)

    knights = [
        action_id
        for action_id in table.promotion_action_ids.tolist()
        if int(table.promotion_indices[action_id]) == KNIGHT_PROMOTION_INDEX
    ]
    for action_id in knights:
        square_pair = int(table.from_to_indices[action_id])
        plain = int((table.from_to_indices == square_pair).nonzero()[0][0])
        torch.testing.assert_close(logits[:, action_id], logits[:, plain])


def test_the_from_to_head_gives_the_other_promotions_their_own_logits() -> None:
    torch.manual_seed(17)
    head = ChessFromToAttentionPolicyHead(32, 16, 8, 8)
    table = CHESS_FROM_TO_ACTION_TABLE
    features = torch.rand((2, 32, 8, 8))

    logits = head(features)

    queens = [
        action_id for action_id in table.promotion_action_ids.tolist() if int(table.promotion_indices[action_id]) == 0
    ]
    knights = [action_id + KNIGHT_PROMOTION_INDEX for action_id in queens]

    assert not torch.allclose(logits[:, queens], logits[:, knights])


@pytest.mark.parametrize(
    'attention_bias',
    (DisabledAttentionBiasConfiguration(), RelativeAttentionBiasConfiguration(), SMOLGEN),
    ids=('disabled', 'relative', 'smolgen'),
)
@pytest.mark.parametrize('policy_head', (FROM_TO_HEAD, DENSE_HEAD), ids=('from_to', 'dense'))
def test_the_exported_inference_model_scripts_and_matches_eager(attention_bias: object, policy_head: object) -> None:
    torch.manual_seed(19)
    model = Network(
        attention_parameters(attention_bias=attention_bias, policy_head=policy_head),
        DEVICE,
        CHESS_NETWORK_DIMENSIONS,
    )
    export = fused_export(model)
    states = chess_states(4)

    scripted = torch.jit.script(export)

    torch.testing.assert_close(scripted(states)[0], export(states)[0])
    torch.testing.assert_close(scripted(states)[1], export(states)[1])


@pytest.mark.parametrize(
    'attention_bias',
    (DisabledAttentionBiasConfiguration(), RelativeAttentionBiasConfiguration(), SMOLGEN),
    ids=('disabled', 'relative', 'smolgen'),
)
def test_generation_zero_calibration_reaches_the_target_on_a_from_to_attention_export(
    attention_bias: object,
) -> None:
    torch.manual_seed(23)
    model = Network(attention_parameters(attention_bias=attention_bias), DEVICE, CHESS_NETWORK_DIMENSIONS)
    export = fused_export(model)
    probe_states = chess_states(64)

    calibrate_bootstrap_policy_prior(export, probe_states)

    shape = measure_policy_prior_shape(export, probe_states)
    assert shape.top3_mass == pytest.approx(BOOTSTRAP_POLICY_PRIOR_TARGET_TOP3_MASS, abs=0.01)


def test_an_uncalibrated_attention_export_is_almost_uniform() -> None:
    torch.manual_seed(29)
    model = Network(attention_parameters(), DEVICE, CHESS_NETWORK_DIMENSIONS)

    shape = measure_policy_prior_shape(fused_export(model), chess_states(64))

    assert shape.top3_mass < 0.2


def test_every_layer_shares_one_generated_bias_template_bank() -> None:
    model = Network(attention_parameters(num_layers=4, attention_bias=SMOLGEN), DEVICE, CHESS_NETWORK_DIMENSIONS)

    banks = {id(block.attention_bias.template_bank) for block in model.backbone}

    assert len(banks) == 1


def test_the_generated_bias_costs_one_template_bank_plus_a_per_layer_compression() -> None:
    two_layers = Network(attention_parameters(num_layers=2, attention_bias=SMOLGEN), DEVICE, CHESS_NETWORK_DIMENSIONS)
    four_layers = Network(attention_parameters(num_layers=4, attention_bias=SMOLGEN), DEVICE, CHESS_NETWORK_DIMENSIONS)
    plain_two = Network(attention_parameters(num_layers=2), DEVICE, CHESS_NETWORK_DIMENSIONS)
    plain_four = Network(attention_parameters(num_layers=4), DEVICE, CHESS_NETWORK_DIMENSIONS)

    def total(model: Network) -> int:
        return sum(parameter.numel() for parameter in model.parameters())

    bank = SMOLGEN.generated_size * BOARD_SQUARE_COUNT * BOARD_SQUARE_COUNT
    per_layer = (total(four_layers) - total(plain_four) - bank) // 4

    assert total(two_layers) - total(plain_two) == bank + 2 * per_layer


def test_the_relative_bias_costs_two_hundred_and_twenty_five_parameters_per_head() -> None:
    biased = Network(
        attention_parameters(num_layers=3, attention_bias=RelativeAttentionBiasConfiguration()),
        DEVICE,
        CHESS_NETWORK_DIMENSIONS,
    )
    plain = Network(attention_parameters(num_layers=3), DEVICE, CHESS_NETWORK_DIMENSIONS)

    def total(model: Network) -> int:
        return sum(parameter.numel() for parameter in model.parameters())

    assert total(biased) - total(plain) == 3 * 4 * 225


def test_a_disabled_bias_leaves_the_attention_call_without_a_mask() -> None:
    model = Network(attention_parameters(), DEVICE, CHESS_NETWORK_DIMENSIONS)

    assert not any(block.uses_attention_bias for block in model.backbone)


def test_a_generated_bias_reaches_the_attention_call() -> None:
    model = Network(attention_parameters(attention_bias=SMOLGEN), DEVICE, CHESS_NETWORK_DIMENSIONS)

    assert all(block.uses_attention_bias for block in model.backbone)
    assert all(isinstance(block.attention_bias, SmolgenAttentionBias) for block in model.backbone)


def test_the_generated_bias_changes_the_trunk_output() -> None:
    torch.manual_seed(31)
    biased = Network(attention_parameters(attention_bias=SMOLGEN), DEVICE, CHESS_NETWORK_DIMENSIONS)
    torch.manual_seed(31)
    plain = Network(attention_parameters(), DEVICE, CHESS_NETWORK_DIMENSIONS)
    states = chess_states(2)

    with torch.no_grad():
        torch.nn.init.normal_(next(iter(biased.backbone)).attention_bias.template_bank.projection.weight, std=0.5)

    assert not torch.allclose(biased.trunk_features(states), plain.trunk_features(states))


def test_model_cost_splits_the_parameters_and_the_multiply_accumulates() -> None:
    model = Network(attention_parameters(), DEVICE, CHESS_NETWORK_DIMENSIONS)

    cost = measure_model_cost(model, batch_size=4)

    assert cost.parameters.total == sum(parameter.numel() for parameter in model.parameters())
    assert cost.multiply_accumulates_per_position.trunk > 0
    assert cost.multiply_accumulates_per_position.policy_head > 0
    assert cost.multiply_accumulates_per_position.value_head > 0


def test_the_reported_multiply_accumulates_do_not_depend_on_the_probe_batch_size() -> None:
    model = Network(attention_parameters(), DEVICE, CHESS_NETWORK_DIMENSIONS)

    assert (
        measure_model_cost(model, batch_size=2).multiply_accumulates_per_position.total
        == measure_model_cost(model, batch_size=8).multiply_accumulates_per_position.total
    )


def test_the_convolutional_trunk_spends_at_most_sixty_four_mac_per_trunk_parameter() -> None:
    model = Network(
        NetworkParams(
            num_layers=4,
            hidden_size=64,
            residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
            policy_head=DENSE_HEAD,
        ),
        DEVICE,
        CHESS_NETWORK_DIMENSIONS,
    )

    cost = measure_model_cost(model)

    # Every trunk parameter is applied once per square, so 64 multiply-accumulates per parameter is the
    # ceiling; normalization scales and biases carry no multiply at all.
    assert 55.0 <= cost.multiply_accumulates_per_position.trunk / cost.parameters.trunk <= 64.0


def test_the_attention_trunk_spends_more_than_sixty_four_mac_per_trunk_parameter() -> None:
    model = Network(attention_parameters(num_layers=4, embedding_size=64), DEVICE, CHESS_NETWORK_DIMENSIONS)

    cost = measure_model_cost(model)

    assert cost.multiply_accumulates_per_position.trunk / cost.parameters.trunk > 64.0


def test_the_attention_score_matrix_is_the_whole_excess_over_sixty_four_mac_per_parameter() -> None:
    layers, embedding = 4, 64
    model = Network(attention_parameters(num_layers=layers, embedding_size=embedding), DEVICE, CHESS_NETWORK_DIMENSIONS)

    cost = measure_model_cost(model)

    # Scoring 64 queries against 64 keys and mixing the values costs two square-count-squared matmuls per
    # layer, and no parameter pays for them.
    score_matrix = layers * 2 * BOARD_SQUARE_COUNT * BOARD_SQUARE_COUNT * embedding
    parameter_bound = cost.multiply_accumulates_per_position.trunk - score_matrix
    assert 55.0 <= parameter_bound / cost.parameters.trunk <= 64.0


def test_the_attention_student_architecture_carries_the_configured_head_and_bias() -> None:
    arguments = replace(
        STUDENT_ARGUMENTS,
        network_kind=NetworkKind.ATTENTION,
        layers=3,
        hidden_size=64,
        heads=4,
        feedforward=96,
        policy_head_kind=PolicyHeadKind.FROM_TO_ATTENTION,
        attention_bias_kind=AttentionBiasKind.SMOLGEN,
    )

    architecture = student_architecture(arguments)

    assert isinstance(architecture, AttentionNetworkParams)
    assert architecture.feedforward_size == 96
    assert isinstance(architecture.policy_head, ChessFromToAttentionPolicyHeadConfiguration)
    assert isinstance(architecture.attention_bias, SmolgenAttentionBiasConfiguration)


def test_the_convolutional_student_architecture_is_unchanged_by_the_attention_knobs() -> None:
    architecture = student_architecture(replace(STUDENT_ARGUMENTS, heads=7, feedforward=999))

    assert isinstance(architecture, NetworkParams)
    assert architecture.hidden_size == STUDENT_ARGUMENTS.hidden_size


def test_the_student_builds_from_its_own_architecture_on_both_trunks() -> None:
    attention = student_architecture(
        replace(
            STUDENT_ARGUMENTS,
            network_kind=NetworkKind.ATTENTION,
            layers=2,
            hidden_size=32,
            heads=4,
            feedforward=64,
            policy_head_kind=PolicyHeadKind.FROM_TO_ATTENTION,
        )
    )
    model = create_model(attention, DEVICE, CHESS_NETWORK_DIMENSIONS)

    policy_logits, _ = model(chess_states(2))

    assert policy_logits.shape == (2, CHESS_NETWORK_DIMENSIONS.actions)


@pytest.mark.parametrize(
    ('step', 'expected'),
    ((200, 0.005), (8_000, 0.005), (9_000, 0.004), (80_000, 0.004), (90_000, 0.003)),
)
def test_the_production_flat_schedule_decays_by_a_factor_of_one_point_six_seven(step: int, expected: float) -> None:
    rate = learning_rate_at(
        step,
        total_steps=100_000,
        peak_learning_rate=0.005,
        warmup_steps=200,
        schedule=LearningRateSchedule.PRODUCTION_FLAT,
    )

    assert rate == pytest.approx(expected)


@pytest.mark.parametrize(
    ('step', 'expected'),
    (
        (200, 0.005),
        (16_000, 0.005),
        (20_000, 0.005 * 0.1 ** (1 / 3)),
        (60_000, 0.005 * 0.1 ** (2 / 3)),
        (90_000, 0.0005),
    ),
)
def test_the_staged_decay_schedule_drops_the_peak_tenfold(step: int, expected: float) -> None:
    rate = learning_rate_at(
        step,
        total_steps=100_000,
        peak_learning_rate=0.005,
        warmup_steps=200,
        schedule=LearningRateSchedule.STAGED_DECAY,
    )

    assert rate == pytest.approx(expected)


def test_the_cosine_floor_schedule_stops_at_the_floor_instead_of_zero() -> None:
    rate = learning_rate_at(
        100_000,
        total_steps=100_000,
        peak_learning_rate=0.005,
        warmup_steps=200,
        schedule=LearningRateSchedule.COSINE_FLOOR,
        floor_fraction=0.1,
    )

    assert rate == pytest.approx(0.0005)


def test_the_staged_decay_schedule_follows_alphazero_when_the_optimizer_is_sgd() -> None:
    rate = learning_rate_at(
        99_000,
        total_steps=100_000,
        peak_learning_rate=0.2,
        warmup_steps=200,
        schedule=LearningRateSchedule.STAGED_DECAY,
        optimizer_kind=OptimizerKind.SGD_MOMENTUM,
    )

    assert rate == pytest.approx(0.2 * ALPHAZERO_SGD_STAGES[-1][1])


@pytest.mark.parametrize(
    ('kind', 'expected'),
    ((OptimizerKind.ADAMW, torch.optim.AdamW), (OptimizerKind.SGD_MOMENTUM, torch.optim.SGD)),
)
def test_the_student_optimizer_starts_at_the_peak_learning_rate(kind: OptimizerKind, expected: type) -> None:
    model = Network(attention_parameters(), DEVICE, CHESS_NETWORK_DIMENSIONS)

    optimizer = create_student_optimizer(model, kind, 0.004)

    assert isinstance(optimizer, expected)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.004)


def test_the_from_to_head_rejects_a_board_that_is_not_chess() -> None:
    with pytest.raises(ValueError, match='from-to attention policy head'):
        Network(attention_parameters(), DEVICE, replace(CHESS_NETWORK_DIMENSIONS, actions=64))
