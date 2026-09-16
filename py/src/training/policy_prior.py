from __future__ import annotations

from pathlib import Path

import torch
from src.training.checkpoint.contracts import BootstrapPolicyPriorRecord, load_checkpoint_manifest
from src.training.network import Network, measure_bootstrap_candidate, temporary_policy_prior_scale


def scheduled_inference_policy_scale(initial_scale: float, generation: int, fade_generations: int) -> float:
    if fade_generations <= 0:
        raise ValueError('Inference-only policy scaling requires a positive fade duration.')
    if generation >= fade_generations:
        return 1.0
    exponent = (fade_generations - generation) / fade_generations
    return max(1.0, initial_scale) ** exponent


def inference_only_policy_prior_record(
    model: Network,
    generation: int,
    save_folder: Path,
    probe_states: torch.Tensor,
    target_top3_mass: float,
    fade_generations: int,
    bootstrap_record: BootstrapPolicyPriorRecord | None,
) -> BootstrapPolicyPriorRecord:
    measurement = measure_bootstrap_candidate(model, probe_states, target_top3_mass)
    initial_record = bootstrap_record
    if initial_record is None and generation > 0:
        initial_record = load_checkpoint_manifest(generation - 1, save_folder).policy_prior_calibration
    if initial_record is None and generation > 0:
        raise ValueError('Inference-only policy scaling requires the previous checkpoint policy record.')
    if initial_record is None:
        initial_scale = measurement.required_policy_scale
    else:
        initial_scale = initial_record.initial_applied_scale or initial_record.applied_scale
    scheduled_scale = scheduled_inference_policy_scale(initial_scale, generation, fade_generations)
    applied_scale = min(scheduled_scale, max(1.0, measurement.required_policy_scale))
    with temporary_policy_prior_scale(model, applied_scale):
        calibrated_shape = measure_bootstrap_candidate(model, probe_states, target_top3_mass).policy_shape
    return BootstrapPolicyPriorRecord(
        candidate_count=initial_record.candidate_count if generation == 0 and initial_record is not None else None,
        selected_candidate_index=(
            initial_record.selected_candidate_index if generation == 0 and initial_record is not None else None
        ),
        selected_candidate_seed=(
            initial_record.selected_candidate_seed if generation == 0 and initial_record is not None else None
        ),
        initial_top1_mass=measurement.policy_shape.top1_mass,
        initial_top3_mass=measurement.policy_shape.top3_mass,
        calibrated_top1_mass=calibrated_shape.top1_mass,
        calibrated_top3_mass=calibrated_shape.top3_mass,
        target_top3_mass=target_top3_mass,
        applied_scale=applied_scale,
        initial_applied_scale=initial_scale,
        mean_wdl_entropy_ratio=measurement.mean_wdl_entropy_ratio,
        mean_absolute_expected_value=measurement.mean_absolute_expected_value,
    )
