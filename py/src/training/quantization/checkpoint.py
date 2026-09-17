from __future__ import annotations

import copy
import time
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
from src.training.checkpoint.persistence import create_model, load_model_state_dict, load_optimizer
from src.training.configuration import BootstrapPolicyScaleApplication, OptimizerConfiguration
from src.training.network import (
    BOOTSTRAP_POLICY_PRIOR_TARGET_TOP3_MASS,
    InferenceNetwork,
    Network,
    NetworkConfiguration,
    calibrate_bootstrap_policy_prior,
    temporary_policy_prior_scale,
)
from src.training.policy_prior import inference_only_policy_prior_record
from src.training.quantization.configuration import (
    QatCheckpointPhase,
    QatFoldingMode,
    QatStateIdentity,
    TensorRtInt8QatConfiguration,
)
from src.training.quantization.runtime import (
    export_float_qat_onnx,
    export_qat_onnx,
    fold_post_activation_batch_norm,
    recalibrate_qat,
    restore_qat_model,
    specialize_float_onnx_batch,
    specialize_qat_onnx_batch,
)
from src.training.targets import AuxiliaryHeadLayout
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.hashing import file_sha256
from src.util.log import log


def _temporary_path(path: Path) -> Path:
    return path.with_name(f'.{path.name}.tmp')


def qat_inference_model_path(generation: int, save_folder: Path) -> Path:
    return save_folder / f'model_{generation}.int8.onnx'


def float_qat_inference_model_path(generation: int, save_folder: Path) -> Path:
    return save_folder / f'model_{generation}.fp16.onnx'


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


def _folded_deployment_copy(model: Network, example_states: torch.Tensor) -> Network:
    copy_started_at = time.perf_counter()
    deployment_model = copy.deepcopy(model)
    copy_seconds = time.perf_counter() - copy_started_at

    fold_started_at = time.perf_counter()
    fold_post_activation_batch_norm(deployment_model)
    fold_seconds = time.perf_counter() - fold_started_at

    def calibration_loop(calibration_model: torch.nn.Module) -> None:
        was_training = calibration_model.training
        calibration_model.eval()
        try:
            with torch.inference_mode():
                calibration_model(example_states)
        finally:
            calibration_model.train(was_training)

    calibration_started_at = time.perf_counter()
    recalibrate_qat(deployment_model, calibration_loop, distributed_sync=False)
    calibration_seconds = time.perf_counter() - calibration_started_at
    log(
        'Prepared folded QAT deployment copy: '
        f'copy={copy_seconds:.3f}s, fold={fold_seconds:.3f}s, recalibration={calibration_seconds:.3f}s.'
    )
    return deployment_model


def save_qat_model_and_optimizer(
    model: Network,
    optimizer: torch.optim.Optimizer,
    generation: int,
    completed_optimizer_steps: int,
    save_folder: Path,
    qat_state: QatStateIdentity,
    example_states: torch.Tensor,
    bootstrap_with_torchscript: bool = False,
    bootstrap_probe_states: torch.Tensor | None = None,
    bootstrap_policy_prior_target_top3_mass: float = BOOTSTRAP_POLICY_PRIOR_TARGET_TOP3_MASS,
    quantization_configuration: TensorRtInt8QatConfiguration | None = None,
    bootstrap_policy_prior: BootstrapPolicyPriorRecord | None = None,
    bootstrap_policy_scale_application: BootstrapPolicyScaleApplication = BootstrapPolicyScaleApplication.TRAINABLE,
    bootstrap_policy_scale_fade_generations: int = 0,
) -> CheckpointReference:
    if qat_state.completed_optimizer_steps != completed_optimizer_steps:
        raise ValueError('QAT state progress must match checkpoint optimizer progress.')
    if generation == 0 and bootstrap_probe_states is None and bootstrap_policy_prior is None:
        raise ValueError('The generation-0 QAT export requires real bootstrap policy probe states.')
    raw_model_path = model_save_path(generation, save_folder)
    raw_optimizer_path = optimizer_save_path(generation, save_folder)
    stored_qat_state_path = qat_state_save_path(generation, save_folder)
    int8_start_generation = (
        1 if quantization_configuration is None else quantization_configuration.int8_self_play_start_generation
    )
    if generation == 0 and bootstrap_with_torchscript:
        inference_path = raw_model_path.with_suffix('.jit.pt')
    elif generation < int8_start_generation:
        inference_path = float_qat_inference_model_path(generation, save_folder)
    else:
        inference_path = qat_inference_model_path(generation, save_folder)
    temporary_model_path = _temporary_path(raw_model_path)
    temporary_optimizer_path = _temporary_path(raw_optimizer_path)
    policy_prior_calibration = bootstrap_policy_prior
    if bootstrap_policy_scale_application is BootstrapPolicyScaleApplication.INFERENCE_ONLY:
        if bootstrap_probe_states is None:
            raise ValueError('Inference-only policy scaling requires policy probe states at every generation.')
        policy_prior_calibration = inference_only_policy_prior_record(
            model,
            generation,
            save_folder,
            bootstrap_probe_states,
            bootstrap_policy_prior_target_top3_mass,
            bootstrap_policy_scale_fade_generations,
            bootstrap_policy_prior,
        )
    if generation == 0:
        if policy_prior_calibration is None:
            assert bootstrap_probe_states is not None
            calibration = calibrate_bootstrap_policy_prior(
                model,
                bootstrap_probe_states,
                bootstrap_policy_prior_target_top3_mass,
            )
            policy_prior_calibration = BootstrapPolicyPriorRecord(
                candidate_count=1,
                selected_candidate_index=0,
                initial_top1_mass=calibration.initial_shape.top1_mass,
                initial_top3_mass=calibration.initial_shape.top3_mass,
                calibrated_top1_mass=calibration.calibrated_shape.top1_mass,
                calibrated_top3_mass=calibration.calibrated_shape.top3_mass,
                target_top3_mass=calibration.target_top3_mass,
                applied_scale=calibration.applied_scale,
                initial_applied_scale=calibration.applied_scale,
            )

    torch.save(model.state_dict(), temporary_model_path)
    torch.save(optimizer.state_dict(), temporary_optimizer_path)
    write_bytes_atomically(stored_qat_state_path, qat_state.path.read_bytes())

    export_model = model
    if (
        quantization_configuration is not None
        and quantization_configuration.folding_mode is QatFoldingMode.DEPLOYMENT_COPY
        and qat_state.phase is QatCheckpointPhase.DEPLOYMENT
    ):
        export_model = _folded_deployment_copy(model, example_states)

    def export_inference_artifact() -> None:
        if generation == 0 and bootstrap_with_torchscript:
            inference_model = _bootstrap_inference_model(export_model)
            torch.jit.save(
                torch.jit.script(inference_model),
                str(inference_path),
                _extra_files={'network.json': inference_model.checkpoint_definition().model_dump_json()},
            )
        elif generation < int8_start_generation:
            export_float_qat_onnx(
                export_model,
                inference_path,
                example_states,
                constant_folding=True,
            )
        else:
            export_qat_onnx(export_model, inference_path, example_states)

    if (
        policy_prior_calibration is not None
        and bootstrap_policy_scale_application is BootstrapPolicyScaleApplication.INFERENCE_ONLY
    ):
        with temporary_policy_prior_scale(export_model, policy_prior_calibration.applied_scale):
            export_inference_artifact()
    else:
        export_inference_artifact()
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


def onnx_inference_checkpoint_for_batch(
    checkpoint: CheckpointReference,
    batch_size: int,
) -> CheckpointReference:
    checkpoint.validate_inference_model()
    if checkpoint.inference_model_path.suffix != '.onnx':
        raise ValueError('A batch-specific inference artifact requires an ONNX inference model.')
    precision = 'fp16' if checkpoint.inference_model_path.name.endswith('.fp16.onnx') else 'int8'
    artifact_path = checkpoint.manifest_path.parent / (
        f'model_{checkpoint.generation}.b{batch_size}-{checkpoint.inference_model_sha256[:16]}.{precision}.onnx'
    )
    if not artifact_path.is_file():
        if precision == 'fp16':
            specialize_float_onnx_batch(checkpoint.inference_model_path, artifact_path, batch_size)
        else:
            specialize_qat_onnx_batch(checkpoint.inference_model_path, artifact_path, batch_size)
    return checkpoint.model_copy(
        update={
            'inference_model_path': artifact_path,
            'inference_model_sha256': file_sha256(artifact_path),
        }
    )
