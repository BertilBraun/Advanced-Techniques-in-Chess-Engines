import hashlib
import shutil
import torch
from os import PathLike
from time import sleep
from pathlib import Path

from src.games.representation import NetworkDimensions
from src.training.checkpoint.contracts import (
    CheckpointManifest,
    CheckpointReference,
    load_checkpoint_manifest,
    load_checkpoint_manifest_path,
)
from src.training.checkpoint.paths import checkpoint_manifest_path, model_save_path, optimizer_save_path
from src.training.configuration import OptimizerType
from src.training.network import Network, NetworkConfiguration, NetworkDefinition
from src.util.atomic_file import write_text_atomically
from src.util.log import LogLevel, log


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f'.{path.name}.tmp')


def create_model(
    args: NetworkConfiguration,
    device: torch.device,
    dimensions: NetworkDimensions,
    auxiliary_output_sizes: tuple[int, ...] = (),
) -> Network:
    model = Network(args, device, dimensions, auxiliary_output_sizes)
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
    auxiliary_output_sizes: tuple[int, ...] = (),
) -> Network:
    model = create_model(args, device, dimensions, auxiliary_output_sizes)
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
    return optimizer


def load_model_and_optimizer(
    generation: int,
    args: NetworkConfiguration,
    device: torch.device,
    save_folder: str | PathLike[str],
    type: OptimizerType,
    dimensions: NetworkDimensions,
    auxiliary_output_sizes: tuple[int, ...] = (),
) -> tuple[Network, torch.optim.Optimizer]:
    manifest_path = checkpoint_manifest_path(generation, save_folder)
    if generation == 0 and not manifest_path.exists():
        model = create_model(args, device, dimensions, auxiliary_output_sizes)
        optimizer = create_optimizer(model, type)
        log('Created a new model and optimizer for generation 0.')
        return model, optimizer
    if generation < 0:
        raise ValueError(f'Checkpoint generation cannot be negative: {generation}')

    manifest = load_checkpoint_manifest(generation, save_folder)
    expected_network = NetworkDefinition(
        architecture=args,
        dimensions=dimensions,
        auxiliary_output_sizes=auxiliary_output_sizes,
    )
    if manifest.network != expected_network:
        raise ValueError('Checkpoint network definition does not match the configured architecture.')
    model = load_model(
        Path(save_folder) / manifest.model_path,
        args,
        device,
        dimensions,
        auxiliary_output_sizes,
    )
    optimizer = load_optimizer(Path(save_folder) / manifest.optimizer_path, model, type, device)
    log(f'Model and optimizer loaded from generation {generation}')
    return model, optimizer


def save_model_and_optimizer(
    model: Network,
    optimizer: torch.optim.Optimizer,
    generation: int,
    save_folder: str | PathLike[str],
) -> None:
    raw_model_path = model_save_path(generation, save_folder)
    raw_optimizer_path = optimizer_save_path(generation, save_folder)
    jit_model_path = raw_model_path.with_suffix('.jit.pt')

    temporary_model_path = _temporary_path(raw_model_path)
    temporary_optimizer_path = _temporary_path(raw_optimizer_path)
    temporary_jit_path = _temporary_path(jit_model_path)

    torch.save(model.state_dict(), temporary_model_path)
    torch.save(optimizer.state_dict(), temporary_optimizer_path)

    # Create a copy of the model, then set that to eval mode, fuse it, and save it as a JIT script
    fused_model = Network(model.network_args, model.device, model.dimensions, model.auxiliary_output_sizes)
    fused_model.load_state_dict(model.state_dict())
    fused_model.auxiliaryHeads = torch.nn.ModuleList()
    fused_model.auxiliary_output_sizes = ()
    fused_model.eval()
    fused_model.fuse_model()

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
        model_sha256=_sha256(raw_model_path),
        optimizer_path=raw_optimizer_path.name,
        optimizer_sha256=_sha256(raw_optimizer_path),
        inference_model_path=jit_model_path.name,
        inference_model_sha256=_sha256(jit_model_path),
    )
    manifest_path = checkpoint_manifest_path(generation, save_folder)
    write_text_atomically(manifest_path, manifest.model_dump_json(indent=2) + '\n')


def import_checkpoint(
    source_manifest_path: Path,
    generation: int,
    destination_folder: Path,
) -> CheckpointReference:
    source_manifest = load_checkpoint_manifest_path(source_manifest_path, generation)
    destination_manifest_path = checkpoint_manifest_path(generation, destination_folder)
    if destination_manifest_path.exists():
        destination_reference = CheckpointReference.load(destination_folder, generation)
        destination_manifest = load_checkpoint_manifest(generation, destination_folder)
        source_hashes = (
            source_manifest.model_sha256,
            source_manifest.optimizer_sha256,
            source_manifest.inference_model_sha256,
        )
        destination_hashes = (
            destination_manifest.model_sha256,
            destination_manifest.optimizer_sha256,
            destination_manifest.inference_model_sha256,
        )
        if destination_hashes != source_hashes:
            raise ValueError('Existing imported checkpoint does not match the configured source checkpoint.')
        return destination_reference

    destination_paths = (
        model_save_path(generation, destination_folder),
        optimizer_save_path(generation, destination_folder),
        model_save_path(generation, destination_folder).with_suffix('.jit.pt'),
    )
    if any(path.exists() for path in destination_paths):
        raise ValueError('Imported checkpoint artifacts exist without their checkpoint manifest.')

    destination_folder.mkdir(parents=True, exist_ok=True)
    source_paths = (
        source_manifest_path.parent / source_manifest.model_path,
        source_manifest_path.parent / source_manifest.optimizer_path,
        source_manifest_path.parent / source_manifest.inference_model_path,
    )
    for source_path, destination_path in zip(source_paths, destination_paths, strict=True):
        temporary_path = _temporary_path(destination_path)
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(destination_path)

    imported_manifest = CheckpointManifest(
        generation=generation,
        network=source_manifest.network,
        model_path=destination_paths[0].name,
        model_sha256=source_manifest.model_sha256,
        optimizer_path=destination_paths[1].name,
        optimizer_sha256=source_manifest.optimizer_sha256,
        inference_model_path=destination_paths[2].name,
        inference_model_sha256=source_manifest.inference_model_sha256,
    )
    write_text_atomically(destination_manifest_path, imported_manifest.model_dump_json(indent=2) + '\n')
    return CheckpointReference.load(destination_folder, generation)


def publish_checkpoint(
    source: CheckpointReference,
    generation: int,
    destination_folder: Path,
) -> CheckpointReference:
    destination_manifest_path = checkpoint_manifest_path(generation, destination_folder)
    source_manifest = load_checkpoint_manifest_path(source.manifest_path, source.generation)
    if destination_manifest_path.exists():
        destination = CheckpointReference.load(destination_folder, generation)
        destination_manifest = load_checkpoint_manifest(generation, destination_folder)
        if (
            destination_manifest.network,
            destination_manifest.model_sha256,
            destination_manifest.optimizer_sha256,
            destination_manifest.inference_model_sha256,
        ) != (
            source_manifest.network,
            source_manifest.model_sha256,
            source_manifest.optimizer_sha256,
            source_manifest.inference_model_sha256,
        ):
            raise ValueError('Existing published checkpoint does not match the progressive model checkpoint.')
        return destination

    destination_paths = (
        model_save_path(generation, destination_folder),
        optimizer_save_path(generation, destination_folder),
        model_save_path(generation, destination_folder).with_suffix('.jit.pt'),
    )
    source_paths = (source.model_path, source.optimizer_path, source.inference_model_path)
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
    )
    write_text_atomically(destination_manifest_path, manifest.model_dump_json(indent=2) + '\n')
    return CheckpointReference.load(destination_folder, generation)
