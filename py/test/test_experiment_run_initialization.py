from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from src.experiment import run
from src.games.chess.configuration import ChessExperimentConfiguration
from src.training.checkpoint.contracts import BootstrapPolicyPriorRecord, load_checkpoint_manifest
from src.training.configuration import BootstrapInitializationConfiguration
from src.training.network import (
    ChessFromToAttentionPolicyHeadConfiguration,
    DisabledResidualContext,
    Network,
    NetworkParams,
    measure_bootstrap_candidate,
)
from src.training.progressive import FixedModelSizingConfiguration, ProgressiveModelDefinition
from src.training.quantization import DisabledTrainingQuantization
from test_helpers.chess_configuration import CHESS_EXPERIMENT
from test_helpers.probe_states import bernoulli_probe_states
from torch import Tensor


def _experiment_with_seed(random_seed: int, candidate_count: int = 1) -> ChessExperimentConfiguration:
    model = ProgressiveModelDefinition(
        model_id='deterministic-initialization-test',
        network=NetworkParams(
            num_layers=1,
            hidden_size=16,
            residual_context=DisabledResidualContext(),
            policy_head=ChessFromToAttentionPolicyHeadConfiguration(key_size=8),
            num_value_channels=1,
            value_fc_size=8,
        ),
    )
    training = CHESS_EXPERIMENT.training.model_copy(
        update={
            'random_seed': random_seed,
            'progressive_model_sizing': FixedModelSizingConfiguration(kind='fixed', model=model),
            'trainer': CHESS_EXPERIMENT.training.trainer.model_copy(
                update={
                    'quantization': DisabledTrainingQuantization(),
                    'bootstrap_initialization': BootstrapInitializationConfiguration(
                        candidate_count=candidate_count,
                        maximum_initial_top1_mass=0.999,
                        minimum_wdl_entropy_ratio=0.001,
                        maximum_absolute_expected_value=1.0,
                        minimum_policy_scale=1e-6,
                        maximum_policy_scale=1e6,
                    ),
                }
            ),
        }
    )
    return CHESS_EXPERIMENT.model_copy(update={'training': training})


def _save_checkpoint_zero(
    experiment: ChessExperimentConfiguration,
    output_path: Path,
) -> tuple[dict[str, Tensor], BootstrapPolicyPriorRecord]:
    output_path.mkdir()
    run._save_random_initial_checkpoint(experiment, output_path, torch.device('cpu'), ())
    state = torch.load(output_path / 'model_0.pt', map_location='cpu', weights_only=True)
    manifest = load_checkpoint_manifest(0, output_path)
    assert manifest.policy_prior_calibration is not None
    return state, manifest.policy_prior_calibration


def _state_dicts_are_equal(first: dict[str, Tensor], second: dict[str, Tensor]) -> bool:
    return first.keys() == second.keys() and all(torch.equal(first[name], second[name]) for name in first)


def test_checkpoint_zero_initialization_is_deterministic_for_the_configured_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_states = bernoulli_probe_states(CHESS_EXPERIMENT.network_dimensions, position_count=16)
    monkeypatch.setattr(run, '_bootstrap_probe_states', lambda _experiment: probe_states)

    first_state, first_calibration = _save_checkpoint_zero(_experiment_with_seed(17), tmp_path / 'first')
    torch.manual_seed(999)
    second_state, second_calibration = _save_checkpoint_zero(_experiment_with_seed(17), tmp_path / 'second')
    different_state, different_calibration = _save_checkpoint_zero(_experiment_with_seed(19), tmp_path / 'different')

    assert _state_dicts_are_equal(first_state, second_state)
    assert first_calibration == second_calibration
    assert not _state_dicts_are_equal(first_state, different_state)
    assert first_calibration != different_calibration


def test_checkpoint_zero_selects_a_deterministic_candidate_and_records_its_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_states = bernoulli_probe_states(CHESS_EXPERIMENT.network_dimensions, position_count=16)
    monkeypatch.setattr(run, '_bootstrap_probe_states', lambda _experiment: probe_states)

    experiment = _experiment_with_seed(101, 4)
    measurements = []
    for candidate_index in range(4):
        torch.manual_seed(101 + candidate_index)
        candidate = Network(
            experiment.training.initial_model.network,
            torch.device('cpu'),
            experiment.network_dimensions,
        )
        measurements.append(
            measure_bootstrap_candidate(
                candidate,
                probe_states,
                experiment.training.trainer.bootstrap_policy_prior_target_top3_mass,
            )
        )
    expected_index = min(
        range(4),
        key=lambda index: abs(math.log(measurements[index].required_policy_scale)),
    )

    first_state, first_record = _save_checkpoint_zero(experiment, tmp_path / 'first')
    second_state, second_record = _save_checkpoint_zero(experiment, tmp_path / 'second')

    assert _state_dicts_are_equal(first_state, second_state)
    assert first_record == second_record
    assert first_record.candidate_count == 4
    assert first_record.selected_candidate_index == expected_index
    assert first_record.selected_candidate_seed == 101 + first_record.selected_candidate_index
    assert first_record.mean_wdl_entropy_ratio is not None
    assert first_record.mean_absolute_expected_value is not None
