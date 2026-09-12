from __future__ import annotations

from pathlib import Path

import torch
from src.games.representation import NetworkDimensions
from src.training.checkpoint.contracts import (
    BootstrapPolicyPriorRecord,
    CheckpointManifest,
    CheckpointReference,
    QatCheckpointRecord,
    load_checkpoint_manifest,
)
from src.training.checkpoint.paths import (
    checkpoint_manifest_path,
    model_save_path,
    optimizer_save_path,
    qat_state_save_path,
)
from src.training.checkpoint.persistence import create_model, load_optimizer
from src.training.configuration import OptimizerConfiguration
from src.training.network import (
    InferenceNetwork,
    Network,
    NetworkConfiguration,
    calibrate_bootstrap_policy_prior,
)
from src.training.quantization.configuration import QatStateIdentity, TensorRtInt8QatConfiguration
from src.training.quantization.runtime import export_qat_onnx, restore_qat_model
from src.training.targets import AuxiliaryHeadLayout
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.hashing import file_sha256


def _temporary_path(path: Path) -> Path:
    return path.with_name(f'.{path.name}.tmp')


def qat_inference_model_path(generation: int, save_folder: Path) -> Path:
    return save_folder / f'model_{generation}.int8.onnx'


def save_qat_model_and_optimizer(
    model: Network,
    optimizer: torch.optim.Optimizer,
    generation: int,
    completed_optimizer_steps: int,
    save_folder: Path,
    qat_state: QatStateIdentity,
    example_states: torch.Tensor,
    bootstrap_probe_states: torch.Tensor | None = None,
) -> CheckpointReference:
    if qat_state.completed_optimizer_steps != completed_optimizer_steps:
        raise ValueError('QAT state progress must match checkpoint optimizer progress.')
    if generation == 0 and bootstrap_probe_states is None:
        raise ValueError('The generation-0 QAT export requires real bootstrap policy probe states.')
    raw_model_path = model_save_path(generation, save_folder)
    raw_optimizer_path = optimizer_save_path(generation, save_folder)
    stored_qat_state_path = qat_state_save_path(generation, save_folder)
    inference_path = qat_inference_model_path(generation, save_folder)
    temporary_model_path = _temporary_path(raw_model_path)
    temporary_optimizer_path = _temporary_path(raw_optimizer_path)
    torch.save(model.state_dict(), temporary_model_path)
    torch.save(optimizer.state_dict(), temporary_optimizer_path)
    write_bytes_atomically(stored_qat_state_path, qat_state.path.read_bytes())
    export_model: torch.nn.Module = model
    policy_prior_calibration = None
    if generation == 0:
        assert bootstrap_probe_states is not None
        inference_model = InferenceNetwork(model)
        calibration = calibrate_bootstrap_policy_prior(inference_model, bootstrap_probe_states)
        export_model = inference_model
        policy_prior_calibration = BootstrapPolicyPriorRecord(
            initial_top1_mass=calibration.initial_shape.top1_mass,
            initial_top3_mass=calibration.initial_shape.top3_mass,
            calibrated_top1_mass=calibration.calibrated_shape.top1_mass,
            calibrated_top3_mass=calibration.calibrated_shape.top3_mass,
            target_top3_mass=calibration.target_top3_mass,
            applied_scale=calibration.applied_scale,
        )
    export_qat_onnx(export_model, inference_path, example_states)
    temporary_model_path.replace(raw_model_path)
    temporary_optimizer_path.replace(raw_optimizer_path)
    manifest = CheckpointManifest(
        generation=generation,
        network=model.checkpoint_definition(),
        model_path=raw_model_path.name,
        model_sha256=file_sha256(raw_model_path),
        optimizer_path=raw_optimizer_path.name,
        optimizer_sha256=file_sha256(raw_optimizer_path),
        inference_model_path=inference_path.name,
        inference_model_sha256=file_sha256(inference_path),
        qat=QatCheckpointRecord(
            phase=qat_state.phase,
            completed_optimizer_steps=completed_optimizer_steps,
            state_path=stored_qat_state_path.name,
            state_sha256=file_sha256(stored_qat_state_path),
        ),
        policy_prior_calibration=policy_prior_calibration,
    )
    write_text_atomically(
        checkpoint_manifest_path(generation, save_folder),
        manifest.model_dump_json(indent=2) + '\n',
    )
    return CheckpointReference.load(save_folder, generation)


def load_qat_model_and_optimizer(
    generation: int,
    network_configuration: NetworkConfiguration,
    optimizer_configuration: OptimizerConfiguration,
    quantization_configuration: TensorRtInt8QatConfiguration,
    device: torch.device,
    save_folder: Path,
    dimensions: NetworkDimensions,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...] = (),
) -> tuple[Network, torch.optim.Optimizer, QatStateIdentity]:
    manifest = load_checkpoint_manifest(generation, save_folder)
    reference = CheckpointReference.from_manifest(save_folder, manifest)
    if reference.qat_state is None:
        raise ValueError('QAT training cannot resume from a checkpoint without QAT state.')
    model = create_model(network_configuration, device, dimensions, auxiliary_heads)
    restored = restore_qat_model(model, reference.qat_state, quantization_configuration)
    weights = torch.load(reference.model_path, map_location=device, weights_only=True)
    restored.model.load_state_dict(weights)
    optimizer = load_optimizer(reference.optimizer_path, restored.model, optimizer_configuration, device)
    return restored.model, optimizer, reference.qat_state
