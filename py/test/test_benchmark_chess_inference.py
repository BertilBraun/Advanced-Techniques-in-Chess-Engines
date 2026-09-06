from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.games.representation import NetworkDimensions
from src.training.network import CHESS_POLICY_PLANE_COUNT, Network
from src.training.targets import build_training_target_layout
from tools.benchmark_chess_inference import (
    _benchmark_models,
    parameter_counts,
    parse_execution_modes,
    parse_positive_integer_list,
)

CONFIGURATION_PATH = Path('configs/production/vast-chess-8gpu-optimal.yaml')
CHESS_PLANE_NETWORK_DIMENSIONS = NetworkDimensions(
    channels=CHESS_NETWORK_DIMENSIONS.channels,
    rows=CHESS_NETWORK_DIMENSIONS.rows,
    columns=CHESS_NETWORK_DIMENSIONS.columns,
    actions=CHESS_POLICY_PLANE_COUNT * 64,
)


@pytest.mark.parametrize(
    ('value', 'expected'),
    (
        ('1', (1,)),
        ('1,16,64,256', (1, 16, 64, 256)),
    ),
)
def test_parse_positive_integer_list(value: str, expected: tuple[int, ...]) -> None:
    assert parse_positive_integer_list(value) == expected


@pytest.mark.parametrize('value', ('', '0', '-1', '1,1', 'one'))
def test_parse_positive_integer_list_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_integer_list(value)


def test_parse_execution_modes() -> None:
    assert tuple(mode.value for mode in parse_execution_modes('eager,scripted,compiled')) == (
        'eager',
        'scripted',
        'compiled',
    )


def test_diagnostic_controls_preserve_historical_backbones_with_direct_policy() -> None:
    models = _benchmark_models(CONFIGURATION_PATH, include_diagnostic_controls=True)

    assert tuple(model.model_id for model in models[:2]) == (
        'control-cnn-10x128-direct-policy',
        'control-attention-5x256-direct-policy',
    )
    assert len(models) == 5


def test_production_progressive_model_parameter_counts_are_derived_directly() -> None:
    configuration = load_chess_experiment_configuration(CONFIGURATION_PATH)
    progressive = configuration.training.progressive_model_sizing
    assert progressive is not None
    target_layout = build_training_target_layout(
        CHESS_PLANE_NETWORK_DIMENSIONS.actions,
        configuration.chess.objective.auxiliary_targets,
    )

    # Re-pinned 2026-09-01: the search-budget head left the network with the stopping rework.
    expected_counts = (
        (1_155_572, 1_123_091, 1_066_240, 50_252, 6_599, 32_481),
        (2_167_188, 2_132_563, 2_073_600, 52_300, 6_663, 34_625),
        (4_562_804, 4_526_035, 4_464_960, 54_348, 6_727, 36_769),
    )

    for model, expected in zip(progressive.models, expected_counts, strict=True):
        network = Network(
            model.network,
            torch.device('cpu'),
            CHESS_PLANE_NETWORK_DIMENSIONS,
            target_layout.auxiliary_heads,
        )
        count = parameter_counts(network)

        assert (
            count.training_total,
            count.inference_total,
            count.backbone,
            count.primary_policy_head,
            count.value_head,
            count.auxiliary_heads,
        ) == expected
        assert count.training_total == sum(parameter.numel() for parameter in network.parameters())
