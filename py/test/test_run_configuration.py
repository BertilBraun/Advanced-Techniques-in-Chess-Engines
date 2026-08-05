from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.experiment.run_configuration import (
    ModelVersionLearningRateStage,
    PiecewiseModelVersionLearningRate,
    ResolvedHardware,
    RunConfiguration,
    apply_run_configuration,
    load_run_configuration,
    validate_run_configuration,
)
from src.settings import TRAINING_ARGS


CONFIGURATION_PATHS = tuple(Path(f'configs/chess-clean-credit-4x4070-v{version}.json') for version in range(6, 14))
LATEST_CONFIGURATION_PATH = CONFIGURATION_PATHS[-1]


def resolved_hardware() -> ResolvedHardware:
    return ResolvedHardware(
        visible_gpu_names=('NVIDIA GeForce RTX 4070 SUPER',) * 4,
        visible_gpu_count=4,
        logical_cpu_count=64,
        total_ram_gib=125.7,
        free_disk_gib=99.9,
    )


@pytest.mark.parametrize('configuration_path', CONFIGURATION_PATHS)
def test_credit_configuration_is_valid_for_quoted_hardware(configuration_path: Path) -> None:
    configuration = load_run_configuration(configuration_path)

    validate_run_configuration(configuration, resolved_hardware())


def test_workload_requires_presentation_credit_configuration() -> None:
    candidate = load_run_configuration(LATEST_CONFIGURATION_PATH).model_dump()
    del candidate['workload']['credit_training']

    with pytest.raises(ValidationError, match='credit_training'):
        RunConfiguration.model_validate(candidate)


@pytest.mark.parametrize(
    ('field_name', 'value'),
    (
        ('iterations', 10),
        ('games_per_iteration', 100),
        ('training_sampling_window', 15),
        ('learning_rate_schedule', ({'start_iteration': 0, 'learning_rate': 0.01},)),
        ('evaluation_every_iterations', 1),
    ),
)
def test_workload_rejects_iteration_lifecycle_fields(field_name: str, value: object) -> None:
    candidate = load_run_configuration(LATEST_CONFIGURATION_PATH).model_dump()
    candidate['workload'][field_name] = value

    with pytest.raises(ValidationError, match=field_name):
        RunConfiguration.model_validate(candidate)


def test_run_configuration_applies_required_credit_schedule_directly() -> None:
    configuration = load_run_configuration(LATEST_CONFIGURATION_PATH)
    arguments = copy.deepcopy(TRAINING_ARGS)

    apply_run_configuration(arguments, configuration)

    parameters = arguments.training.credit_training
    assert parameters.optimizer_steps_per_quantum == 500
    assert parameters.maximum_optimizer_steps == 500_000
    assert parameters.replay_ratio == 8
    assert parameters.replay_capacity_for_model_version(0) == 200_000
    assert parameters.replay_capacity_for_model_version(100) == 1_350_000
    assert parameters.replay_capacity_for_model_version(200) == 2_500_000
    assert arguments.training.global_batch_size == 2_048
    assert arguments.training.local_batch_size == 512
    assert arguments.cluster.trainer_ddp_device_ids == (3, 2, 1, 0)


def test_evaluation_configuration_owns_workload_schedule_and_protocol() -> None:
    configuration = load_run_configuration(LATEST_CONFIGURATION_PATH)
    arguments = copy.deepcopy(TRAINING_ARGS)

    apply_run_configuration(arguments, configuration)

    assert configuration.evaluation.games == 100
    assert configuration.evaluation.schedule.interval_optimizer_steps == 1_500
    assert configuration.evaluation.protocol.opening_suite_path.endswith('main-monitoring-openings-50.tsv')
    assert arguments.evaluation_schedule.interval_optimizer_steps == 1_500


@pytest.mark.parametrize(
    ('owner', 'field_name', 'value'),
    (
        ('workload', 'evaluation_games', 100),
        ('credit_training', 'evaluation_timeout_seconds', 7_200),
    ),
)
def test_evaluation_fields_are_rejected_outside_evaluation_configuration(
    owner: str,
    field_name: str,
    value: object,
) -> None:
    candidate = load_run_configuration(LATEST_CONFIGURATION_PATH).model_dump()
    target = candidate['workload'] if owner == 'workload' else candidate['workload']['credit_training']
    target[field_name] = value

    with pytest.raises(ValidationError, match=field_name):
        RunConfiguration.model_validate(candidate)


def test_model_version_learning_rate_uses_latest_started_stage() -> None:
    schedule = PiecewiseModelVersionLearningRate(
        (
            ModelVersionLearningRateStage(start_model_version=0, learning_rate=0.005),
            ModelVersionLearningRateStage(start_model_version=6, learning_rate=0.002),
        ),
        optimizer_steps_per_model_version=10,
    )

    assert schedule(59, 'adamw') == pytest.approx(0.005)
    assert schedule(60, 'adamw') == pytest.approx(0.002)
    assert pickle.loads(pickle.dumps(schedule))(60, 'adamw') == pytest.approx(0.002)


@pytest.mark.parametrize('model_versions', ((100,), (0, 100, 100), (0, 200, 100)))
def test_credit_learning_rate_schedule_requires_zero_based_increasing_model_versions(
    model_versions: tuple[int, ...],
) -> None:
    candidate = load_run_configuration(LATEST_CONFIGURATION_PATH).model_dump()
    candidate['workload']['credit_training']['learning_rate_schedule'] = tuple(
        {'start_model_version': model_version, 'learning_rate': 0.002} for model_version in model_versions
    )

    with pytest.raises(ValidationError, match='learning-rate'):
        RunConfiguration.model_validate(candidate)


def test_credit_training_allows_pausing_a_subset_of_self_play_workers() -> None:
    configuration = load_run_configuration(LATEST_CONFIGURATION_PATH)
    topology = configuration.topology.model_copy(
        update={'self_play_processes_per_device_during_training': (1, 1, 1, 1)}
    )
    candidate = configuration.model_copy(update={'topology': topology})
    arguments = copy.deepcopy(TRAINING_ARGS)

    apply_run_configuration(arguments, candidate)

    assert arguments.cluster.self_play_node_ids_to_pause_during_training == (
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        13,
        14,
        15,
    )
