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
    load_checkpoint_manifest_path,
)
from src.training.checkpoint.paths import (
    checkpoint_manifest_path,
    model_save_path,
    optimizer_save_path,
    qat_state_save_path,
)
from src.training.checkpoint.persistence import create_model, load_model_state_dict, load_optimizer
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


def _bootstrap_inference_model(model: Network) -> InferenceNetwork:
    plain_model = Network(
        model.network_args,
        next(model.parameters()).device,
        model.dimensions,
        model.auxiliary_heads,
    )
    qat_weights = model.state_dict()
    plain_model.load_state_dict({name: qat_weights[name] for name in plain_model.state_dict()})
    inference_model = InferenceNetwork(plain_model)
    inference_model.eval()
    inference_model.fuse_model()
    return inference_model


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
    inference_path = (
        raw_model_path.with_suffix('.jit.pt') if generation == 0 else qat_inference_model_path(generation, save_folder)
    )
    temporary_model_path = _temporary_path(raw_model_path)
    temporary_optimizer_path = _temporary_path(raw_optimizer_path)
    torch.save(model.state_dict(), temporary_model_path)
    torch.save(optimizer.state_dict(), temporary_optimizer_path)
    write_bytes_atomically(stored_qat_state_path, qat_state.path.read_bytes())
    policy_prior_calibration = None
    if generation == 0:
        assert bootstrap_probe_states is not None
        inference_model = _bootstrap_inference_model(model)
        calibration = calibrate_bootstrap_policy_prior(inference_model, bootstrap_probe_states)
        policy_prior_calibration = BootstrapPolicyPriorRecord(
            initial_top1_mass=calibration.initial_shape.top1_mass,
            initial_top3_mass=calibration.initial_shape.top3_mass,
            calibrated_top1_mass=calibration.calibrated_shape.top1_mass,
            calibrated_top3_mass=calibration.calibrated_shape.top3_mass,
            target_top3_mass=calibration.target_top3_mass,
            applied_scale=calibration.applied_scale,
        )
        torch.jit.save(
            torch.jit.script(inference_model),
            str(inference_path),
            _extra_files={'network.json': inference_model.checkpoint_definition().model_dump_json()},
        )
    else:
        export_qat_onnx(model, inference_path, example_states)
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
    load_model_state_dict(restored.model, weights, reference.model_path)
    optimizer = load_optimizer(reference.optimizer_path, restored.model, optimizer_configuration, device)
    return restored.model, optimizer, reference.qat_state


def qat_inference_checkpoint_for_batch(
    checkpoint: CheckpointReference,
    batch_size: int,
    network_configuration: NetworkConfiguration,
    optimizer_configuration: OptimizerConfiguration,
    quantization_configuration: TensorRtInt8QatConfiguration,
    device: torch.device,
    dimensions: NetworkDimensions,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...] = (),
) -> CheckpointReference:
    if batch_size <= 0:
        raise ValueError('QAT inference batch size must be positive.')
    manifest = load_checkpoint_manifest_path(checkpoint.manifest_path, checkpoint.generation)
    if manifest.qat is None:
        raise ValueError('A batch-specific QAT inference artifact requires a QAT checkpoint.')
    if checkpoint.generation == 0:
        raise ValueError('Generation-zero QAT inference uses the TorchScript bootstrap artifact.')
    artifact_path = checkpoint.manifest_path.parent / (
        f'model_{checkpoint.generation}.int8-b{batch_size}-{checkpoint.inference_model_sha256[:16]}.onnx'
    )
    if not artifact_path.is_file():
        model, _, _ = load_qat_model_and_optimizer(
            checkpoint.generation,
            network_configuration,
            optimizer_configuration,
            quantization_configuration,
            device,
            checkpoint.manifest_path.parent,
            dimensions,
            auxiliary_heads,
        )
        example_states = torch.zeros(
            (batch_size, dimensions.channels, dimensions.rows, dimensions.columns),
            device=device,
        )
        export_qat_onnx(model, artifact_path, example_states)
    return checkpoint.model_copy(
        update={
            'inference_model_path': artifact_path,
            'inference_model_sha256': file_sha256(artifact_path),
        }
    )
