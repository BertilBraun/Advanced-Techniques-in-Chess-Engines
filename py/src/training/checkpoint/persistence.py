from __future__ import annotations

import shutil
from os import PathLike
from pathlib import Path
from time import sleep

import torch
from src.games.representation import NetworkDimensions
from src.training.checkpoint.contracts import (
    BootstrapPolicyPriorRecord,
    CheckpointManifest,
    CheckpointReference,
    load_checkpoint_manifest,
    load_checkpoint_manifest_path,
)
from src.training.checkpoint.paths import checkpoint_manifest_path, model_save_path, optimizer_save_path
from src.training.configuration import OptimizerType
from src.training.network import (
    InferenceNetwork,
    Network,
    NetworkConfiguration,
    NetworkDefinition,
    calibrate_bootstrap_policy_prior,
)
from src.training.targets import AuxiliaryHeadLayout
from src.util.atomic_file import write_text_atomically
from src.util.hashing import file_sha256
from src.util.log import LogLevel, log


def _temporary_path(path: Path) -> Path:
    return path.with_name(f'.{path.name}.tmp')


def create_model(
    args: NetworkConfiguration,
    device: torch.device,
    dimensions: NetworkDimensions,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...] = (),
) -> Network:
    model = Network(args, device, dimensions, auxiliary_heads)
    return model


def create_optimizer(model: Network, type: OptimizerType) -> torch.optim.Optimizer:
    if type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001, nesterov=True)
    elif type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True, eps=1e-5)
    raise ValueError(f'Optimizer type {type} not supported. Supported types: adamw, sgd')


def load_model(
    path: str | PathLike[str],
    args: NetworkConfiguration,
    device: torch.device,
    dimensions: NetworkDimensions,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...] = (),
) -> Network:
    model = create_model(args, device, dimensions, auxiliary_heads)
    try:
        for _ in range(5):
            try:
                data = torch.load(path, map_location=device, weights_only=True)
                break
            except EOFError:
                sleep(1)
        else:
            raise ValueError(f'Model remained incomplete after five attempts: {path}')

    except FileNotFoundError as error:
        raise ValueError(f'Model does not exist: {path}') from error

    try:
        model.load_state_dict(data)
    except RuntimeError:
        # check if any key contains "_orig_mod." if so, try to load without it, else try to load
        contains_org = any('_orig_mod.' in key for key in data.keys())

        if contains_org:
            assert all('_orig_mod.' in key for key in data.keys()), 'Some keys contain "_orig_mod." and some do not'

            log(f'Could not load model from: {path}, trying to load without compilation')
            # replace all key prefixes or "_orig_mod." with ""

            data = {key.replace('_orig_mod.', ''): value for key, value in data.items()}
            model.load_state_dict(data)
        else:
            assert all('_orig_mod.' not in key for key in data.keys()), 'Some keys contain "_orig_mod." and some do not'
            log(f'Could not load model from: {path}, trying to load with compilation')

            data = {f'_orig_mod.{key}': value for key, value in data.items()}

        try:
            model.load_state_dict(data)
        except RuntimeError:
            log(f'Could not load model from: {path}')
            raise
    log(f'Model loaded from: {path}', level=LogLevel.DEBUG)
    return model


def load_optimizer(
    path: str | PathLike[str],
    model: Network,
    type: OptimizerType,
    device: torch.device,
) -> torch.optim.Optimizer:
    optimizer = create_optimizer(model, type)
    try:
        for _ in range(5):
            try:
                data = torch.load(path, weights_only=True, map_location=device)
                break
            except EOFError:
                sleep(1)
        else:
            log(f'Could not load optimizer from: {path}')
            raise FileNotFoundError

    except FileNotFoundError:
        log(f'No optimizer found for: {path}')
        raise

    optimizer.load_state_dict(data)
    # map_location puts AdamW step counters on the GPU; foreach AdamW then syncs once per parameter per
    # optimizer step reading them back. Fresh optimizers keep steps on CPU, so restore that layout.
    for state in optimizer.state.values():
        step = state.get('step')
        if isinstance(step, torch.Tensor) and step.device.type != 'cpu':
            state['step'] = step.cpu()
    return optimizer


def load_model_and_optimizer(
    generation: int,
    args: NetworkConfiguration,
    device: torch.device,
    save_folder: str | PathLike[str],
    type: OptimizerType,
    dimensions: NetworkDimensions,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...] = (),
) -> tuple[Network, torch.optim.Optimizer]:
    manifest_path = checkpoint_manifest_path(generation, save_folder)
    if generation == 0 and not manifest_path.exists():
        model = create_model(args, device, dimensions, auxiliary_heads)
        optimizer = create_optimizer(model, type)
        log('Created a new model and optimizer for generation 0.')
        return model, optimizer
    if generation < 0:
        raise ValueError(f'Checkpoint generation cannot be negative: {generation}')

    manifest = load_checkpoint_manifest(generation, save_folder)
    expected_network = NetworkDefinition(
        architecture=args,
        dimensions=dimensions,
        auxiliary_heads=auxiliary_heads,
    )
    if manifest.network != expected_network:
        raise ValueError('Checkpoint network definition does not match the configured architecture.')
    model = load_model(
        Path(save_folder) / manifest.model_path,
        args,
        device,
        dimensions,
        auxiliary_heads,
    )
    optimizer = load_optimizer(Path(save_folder) / manifest.optimizer_path, model, type, device)
    log(f'Model and optimizer loaded from generation {generation}')
    return model, optimizer


def save_model_and_optimizer(
    model: Network,
    optimizer: torch.optim.Optimizer,
    generation: int,
    save_folder: str | PathLike[str],
    bootstrap_probe_states: torch.Tensor | None = None,
) -> None:
    raw_model_path = model_save_path(generation, save_folder)
    raw_optimizer_path = optimizer_save_path(generation, save_folder)
    jit_model_path = raw_model_path.with_suffix('.jit.pt')

    temporary_model_path = _temporary_path(raw_model_path)
    temporary_optimizer_path = _temporary_path(raw_optimizer_path)
    temporary_jit_path = _temporary_path(jit_model_path)

    torch.save(model.state_dict(), temporary_model_path)
    torch.save(optimizer.state_dict(), temporary_optimizer_path)

    fused_model = InferenceNetwork(model)
    fused_model.eval()
    fused_model.fuse_model()
    policy_prior_calibration: BootstrapPolicyPriorRecord | None = None
    if generation == 0:
        if bootstrap_probe_states is None:
            raise ValueError(
                'The generation-0 export requires real probe positions from the evaluation dataset '
                'to calibrate the bootstrap policy prior.'
            )
        calibration = calibrate_bootstrap_policy_prior(fused_model, bootstrap_probe_states)
        log(
            f'Calibrated the generation-0 policy prior: top-1 mass '
            f'{calibration.initial_shape.top1_mass:.3f} -> {calibration.calibrated_shape.top1_mass:.3f}, '
            f'top-3 mass {calibration.initial_shape.top3_mass:.3f} -> {calibration.calibrated_shape.top3_mass:.3f} '
            f'(target {calibration.target_top3_mass}), applied scale {calibration.applied_scale:.4g}.'
        )
        policy_prior_calibration = BootstrapPolicyPriorRecord(
            initial_top1_mass=calibration.initial_shape.top1_mass,
            initial_top3_mass=calibration.initial_shape.top3_mass,
            calibrated_top1_mass=calibration.calibrated_shape.top1_mass,
            calibrated_top3_mass=calibration.calibrated_shape.top3_mass,
            target_top3_mass=calibration.target_top3_mass,
            applied_scale=calibration.applied_scale,
        )

    torch.jit.save(
        torch.jit.script(fused_model),
        str(temporary_jit_path),
        _extra_files={'network.json': fused_model.checkpoint_definition().model_dump_json()},
    )

    temporary_model_path.replace(raw_model_path)
    temporary_optimizer_path.replace(raw_optimizer_path)
    temporary_jit_path.replace(jit_model_path)

    manifest = CheckpointManifest(
        generation=generation,
        network=model.checkpoint_definition(),
        model_path=raw_model_path.name,
        model_sha256=file_sha256(raw_model_path),
        optimizer_path=raw_optimizer_path.name,
        optimizer_sha256=file_sha256(raw_optimizer_path),
        inference_model_path=jit_model_path.name,
        inference_model_sha256=file_sha256(jit_model_path),
        policy_prior_calibration=policy_prior_calibration,
    )
    manifest_path = checkpoint_manifest_path(generation, save_folder)
    write_text_atomically(manifest_path, manifest.model_dump_json(indent=2) + '\n')


def _checkpoint_identity(manifest: CheckpointManifest) -> tuple[NetworkDefinition, str, str, str]:
    return (
        manifest.network,
        manifest.model_sha256,
        manifest.optimizer_sha256,
        manifest.inference_model_sha256,
    )


def _copy_checkpoint(
    source_manifest: CheckpointManifest,
    source_paths: tuple[Path, Path, Path],
    generation: int,
    destination_folder: Path,
    mismatch_message: str,
) -> CheckpointReference:
    destination_manifest_path = checkpoint_manifest_path(generation, destination_folder)
    if destination_manifest_path.exists():
        destination = CheckpointReference.load(destination_folder, generation)
        destination_manifest = load_checkpoint_manifest(generation, destination_folder)
        if _checkpoint_identity(destination_manifest) != _checkpoint_identity(source_manifest):
            raise ValueError(mismatch_message)
        return destination

    destination_paths = (
        model_save_path(generation, destination_folder),
        optimizer_save_path(generation, destination_folder),
        model_save_path(generation, destination_folder).with_suffix('.jit.pt'),
    )
    if any(path.exists() for path in destination_paths):
        raise ValueError('Checkpoint artifacts exist without their checkpoint manifest.')

    destination_folder.mkdir(parents=True, exist_ok=True)
    for source_path, destination_path in zip(source_paths, destination_paths, strict=True):
        temporary_path = _temporary_path(destination_path)
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(destination_path)

    manifest = CheckpointManifest(
        generation=generation,
        network=source_manifest.network,
        model_path=destination_paths[0].name,
        model_sha256=source_manifest.model_sha256,
        optimizer_path=destination_paths[1].name,
        optimizer_sha256=source_manifest.optimizer_sha256,
        inference_model_path=destination_paths[2].name,
        inference_model_sha256=source_manifest.inference_model_sha256,
        policy_prior_calibration=source_manifest.policy_prior_calibration,
    )
    write_text_atomically(destination_manifest_path, manifest.model_dump_json(indent=2) + '\n')
    return CheckpointReference.load(destination_folder, generation)


def import_checkpoint(
    source_manifest_path: Path,
    generation: int,
    destination_folder: Path,
) -> CheckpointReference:
    source_manifest = load_checkpoint_manifest_path(source_manifest_path, generation)
    source_paths = (
        source_manifest_path.parent / source_manifest.model_path,
        source_manifest_path.parent / source_manifest.optimizer_path,
        source_manifest_path.parent / source_manifest.inference_model_path,
    )
    return _copy_checkpoint(
        source_manifest,
        source_paths,
        generation,
        destination_folder,
        'Existing imported checkpoint does not match the configured source checkpoint.',
    )


def publish_checkpoint(
    source: CheckpointReference,
    generation: int,
    destination_folder: Path,
) -> CheckpointReference:
    source_manifest = load_checkpoint_manifest_path(source.manifest_path, source.generation)
    source_paths = (source.model_path, source.optimizer_path, source.inference_model_path)
    return _copy_checkpoint(
        source_manifest,
        source_paths,
        generation,
        destination_folder,
        'Existing published checkpoint does not match the progressive model checkpoint.',
    )
