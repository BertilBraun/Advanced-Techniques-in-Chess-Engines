from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from src.games.representation import NetworkDimensions
from src.training.checkpoint import BootstrapPolicyPriorRecord
from src.training.checkpoint.persistence import create_model
from src.training.configuration import BootstrapInitializationConfiguration, BootstrapPolicyScaleApplication
from src.training.network import (
    BootstrapCandidateMeasurement,
    Network,
    NetworkConfiguration,
    calibrate_bootstrap_policy_prior,
    measure_bootstrap_candidate,
)
from src.training.targets import AuxiliaryHeadLayout
from src.util.log import log


@dataclass(frozen=True)
class SelectedBootstrapModel:
    model: Network
    record: BootstrapPolicyPriorRecord


def _candidate_is_healthy(
    constraints: BootstrapInitializationConfiguration,
    measurement: BootstrapCandidateMeasurement,
) -> bool:
    return (
        measurement.policy_shape.top1_mass <= constraints.maximum_initial_top1_mass
        and measurement.mean_wdl_entropy_ratio >= constraints.minimum_wdl_entropy_ratio
        and measurement.mean_absolute_expected_value <= constraints.maximum_absolute_expected_value
        and constraints.minimum_policy_scale <= measurement.required_policy_scale <= constraints.maximum_policy_scale
    )


def select_bootstrap_model(
    network: NetworkConfiguration,
    device: torch.device,
    dimensions: NetworkDimensions,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...],
    probe_states: torch.Tensor,
    constraints: BootstrapInitializationConfiguration,
    target_top3_mass: float,
    base_seed: int,
) -> SelectedBootstrapModel:
    selected_index: int | None = None
    selected_seed: int | None = None
    selected_measurement: BootstrapCandidateMeasurement | None = None
    selected_distance = math.inf
    for candidate_index in range(constraints.candidate_count):
        candidate_seed = base_seed + candidate_index
        torch.manual_seed(candidate_seed)
        candidate = create_model(network, device, dimensions, auxiliary_heads)
        measurement = measure_bootstrap_candidate(candidate, probe_states, target_top3_mass)
        healthy = _candidate_is_healthy(constraints, measurement)
        log(
            f'Bootstrap candidate {candidate_index + 1}/{constraints.candidate_count} seed {candidate_seed}: '
            f'top-1 {measurement.policy_shape.top1_mass:.4f}, '
            f'top-3 {measurement.policy_shape.top3_mass:.4f}, '
            f'policy scale {measurement.required_policy_scale:.4g}, '
            f'WDL entropy ratio {measurement.mean_wdl_entropy_ratio:.4f}, '
            f'absolute expected value {measurement.mean_absolute_expected_value:.4f}, '
            f'{"healthy" if healthy else "rejected"}.'
        )
        if not healthy:
            continue
        distance = abs(math.log(measurement.required_policy_scale))
        if distance < selected_distance:
            selected_index = candidate_index
            selected_seed = candidate_seed
            selected_measurement = measurement
            selected_distance = distance
    if selected_index is None or selected_seed is None or selected_measurement is None:
        raise ValueError(f'None of the {constraints.candidate_count} bootstrap initialization candidates was healthy.')

    torch.manual_seed(selected_seed)
    selected_model = create_model(network, device, dimensions, auxiliary_heads)
    match constraints.policy_scale_application:
        case BootstrapPolicyScaleApplication.TRAINABLE:
            calibration_model = selected_model
        case BootstrapPolicyScaleApplication.INFERENCE_ONLY:
            calibration_model = copy.deepcopy(selected_model)
    calibration = calibrate_bootstrap_policy_prior(calibration_model, probe_states, target_top3_mass)
    if not math.isclose(calibration.applied_scale, selected_measurement.required_policy_scale, rel_tol=1e-9):
        raise AssertionError('Recreated bootstrap candidate did not reproduce its measured policy scale.')
    log(
        f'Selected bootstrap candidate {selected_index + 1}/{constraints.candidate_count} seed {selected_seed} '
        f'with {constraints.policy_scale_application.value} policy scale {calibration.applied_scale:.4g}.'
    )
    return SelectedBootstrapModel(
        model=selected_model,
        record=BootstrapPolicyPriorRecord(
            candidate_count=constraints.candidate_count,
            selected_candidate_index=selected_index,
            selected_candidate_seed=selected_seed,
            initial_top1_mass=calibration.initial_shape.top1_mass,
            initial_top3_mass=calibration.initial_shape.top3_mass,
            calibrated_top1_mass=calibration.calibrated_shape.top1_mass,
            calibrated_top3_mass=calibration.calibrated_shape.top3_mass,
            target_top3_mass=calibration.target_top3_mass,
            applied_scale=calibration.applied_scale,
            mean_wdl_entropy_ratio=selected_measurement.mean_wdl_entropy_ratio,
            mean_absolute_expected_value=selected_measurement.mean_absolute_expected_value,
        ),
    )
