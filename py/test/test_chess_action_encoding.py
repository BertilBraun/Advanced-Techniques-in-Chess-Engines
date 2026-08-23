from __future__ import annotations

import pytest
import torch
from src.games.chess.contract import CHESS_ACTION_SIZE, CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.training.network import (
    SMALL_OUTPUT_INITIALIZATION_STD,
    Chess76PlaneDirectPolicyHeadConfiguration,
    DensePolicyHeadConfiguration,
    DisabledResidualContext,
    Network,
    NetworkParams,
    PolicyHeadConfiguration,
)
from src.training.targets import AuxiliaryHeadLayout, LegalMovesHeadLayout, NextPolicyHeadLayout
from torch import nn

REDUCED_ACTION_SIZE = 1880
TRUNK_CHANNELS = 128
BOTTLENECK_CHANNELS = 2
DENSE_HEAD_PARAMETER_COUNT = 242_780


def _chess_network(
    policy_head: PolicyHeadConfiguration,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...] = (),
) -> Network:
    return Network(
        NetworkParams(
            num_layers=1,
            hidden_size=TRUNK_CHANNELS,
            residual_context=DisabledResidualContext(),
            policy_head=policy_head,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
        auxiliary_heads,
    )


def test_chess_contract_exposes_the_reduced_action_space() -> None:
    assert CHESS_ACTION_SIZE == REDUCED_ACTION_SIZE
    assert CHESS_STATE_CONTRACT.action_size == CHESS_ACTION_SIZE
    assert CHESS_NETWORK_DIMENSIONS.actions == CHESS_ACTION_SIZE


@pytest.mark.native
def test_native_chess_action_size_matches_the_python_contract() -> None:
    native = pytest.importorskip('AlphaZeroCpp')
    assert native.CHESS_ACTION_SIZE == CHESS_ACTION_SIZE


@pytest.mark.native
def test_native_chess_mirror_action_id_is_an_involution_over_the_action_space() -> None:
    native = pytest.importorskip('AlphaZeroCpp')
    for action_id in range(native.CHESS_ACTION_SIZE):
        mirrored = native.mirror_chess_action_id(action_id)
        assert 0 <= mirrored < native.CHESS_ACTION_SIZE
        assert native.mirror_chess_action_id(mirrored) == action_id


@pytest.mark.native
def test_native_chess_mirror_action_id_rejects_ids_outside_the_action_space() -> None:
    native = pytest.importorskip('AlphaZeroCpp')
    with pytest.raises(ValueError):
        native.mirror_chess_action_id(native.CHESS_ACTION_SIZE)


@pytest.mark.native
def test_native_chess_legal_actions_round_trip_inside_the_reduced_action_space() -> None:
    native = pytest.importorskip('AlphaZeroCpp')
    position = native.ChessPosition()
    action_ids = position.legal_actions()
    assert len(action_ids) == 20
    assert len(set(action_ids)) == len(action_ids)
    for action_id in action_ids:
        assert 0 <= action_id < CHESS_ACTION_SIZE
        assert position.action_id_from_uci(position.action_uci(action_id)) == action_id


def test_plane_policy_head_rejects_the_reduced_chess_action_space() -> None:
    with pytest.raises(ValueError, match='reduced chess action encoding'):
        _chess_network(Chess76PlaneDirectPolicyHeadConfiguration())


def test_dense_policy_head_bottleneck_sizes_the_reduced_chess_action_space() -> None:
    torch.manual_seed(5)
    network = _chess_network(DensePolicyHeadConfiguration(channels=BOTTLENECK_CHANNELS))
    network.eval()

    logits, _ = network.logit_forward(torch.randn((2, 29, 8, 8)))
    head = network.policy_head
    assert isinstance(head, nn.Sequential)
    bottleneck = head[0]
    output_projection = head[-1]
    assert isinstance(bottleneck, nn.Conv2d)
    assert isinstance(output_projection, nn.Linear)

    assert logits.shape == (2, CHESS_ACTION_SIZE)
    assert bottleneck.out_channels == BOTTLENECK_CHANNELS
    assert output_projection.in_features == BOTTLENECK_CHANNELS * 8 * 8
    assert output_projection.out_features == CHESS_ACTION_SIZE
    assert sum(parameter.numel() for parameter in head.parameters()) == DENSE_HEAD_PARAMETER_COUNT
    assert float(output_projection.weight.detach().std()) == pytest.approx(SMALL_OUTPUT_INITIALIZATION_STD, rel=0.1)


@pytest.mark.parametrize('bottleneck_channels', (2, 4))
def test_dense_policy_head_bottleneck_channels_are_configuration_selectable(bottleneck_channels: int) -> None:
    network = _chess_network(DensePolicyHeadConfiguration(channels=bottleneck_channels))
    assert network.policy_head[0].out_channels == bottleneck_channels
    assert network.policy_head[-1].in_features == bottleneck_channels * 8 * 8


def test_policy_shaped_auxiliary_heads_follow_the_reduced_chess_action_space() -> None:
    network = _chess_network(
        DensePolicyHeadConfiguration(channels=BOTTLENECK_CHANNELS),
        (
            NextPolicyHeadLayout(kind='next_policy', action_size=CHESS_ACTION_SIZE, ply_offset=1),
            LegalMovesHeadLayout(kind='legal_moves', action_size=CHESS_ACTION_SIZE),
        ),
    )

    output = network.training_output(torch.randn((2, 29, 8, 8)))

    assert output.policy_logits.shape == (2, CHESS_ACTION_SIZE)
    assert tuple(logits.shape for logits in output.auxiliary_logits) == (
        (2, CHESS_ACTION_SIZE),
        (2, CHESS_ACTION_SIZE),
    )


def test_policy_shaped_auxiliary_heads_reject_a_mismatched_action_space() -> None:
    with pytest.raises(ValueError, match='Next-policy action space'):
        _chess_network(
            DensePolicyHeadConfiguration(channels=BOTTLENECK_CHANNELS),
            (NextPolicyHeadLayout(kind='next_policy', action_size=CHESS_ACTION_SIZE + 1, ply_offset=1),),
        )
