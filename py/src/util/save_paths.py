import hashlib
import os
import torch
from os import PathLike
from time import sleep
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.Network import Network
from src.train.TrainingArgs import NetworkParams, OptimizerType
from src.util.log import LogLevel, log


class ReplayFileReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    path: str
    size_bytes: int


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    iteration: int
    model_path: str
    model_sha256: str
    optimizer_path: str
    optimizer_sha256: str
    jit_model_path: str
    jit_model_sha256: str
    replay_files: tuple[ReplayFileReference, ...]


def model_save_path(iteration: int, save_folder: str | PathLike) -> Path:
    path = Path(save_folder) / f'model_{iteration}.pt'
    if not path.exists():
        os.makedirs(path.parent, exist_ok=True)
    return path


def optimizer_save_path(iteration: int, save_folder: str | PathLike) -> Path:
    path = Path(save_folder) / f'optimizer_{iteration}.pt'
    if not path.exists():
        os.makedirs(path.parent, exist_ok=True)
    return path


def checkpoint_manifest_path(iteration: int, save_folder: str | PathLike) -> Path:
    return Path(save_folder) / f'checkpoint_{iteration}.json'


def inference_model_path(path: str | PathLike) -> Path:
    model_path = Path(path)
    if model_path.name.endswith('.jit.pt'):
        return model_path
    if model_path.suffix == '.pt':
        return model_path.with_suffix('.jit.pt')
    raise ValueError(f'Model path must end in .pt or .jit.pt: {model_path}')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f'.{path.name}.tmp')


def _replay_file_references(save_folder: str | PathLike) -> tuple[ReplayFileReference, ...]:
    root = Path(save_folder)
    references: list[ReplayFileReference] = []
    for replay_path in sorted(root.glob('memory_*/*.hdf5')):
        references.append(
            ReplayFileReference(
                path=replay_path.relative_to(root).as_posix(),
                size_bytes=replay_path.stat().st_size,
            )
        )
    return tuple(references)


def load_checkpoint_manifest(
    iteration: int,
    save_folder: str | PathLike,
) -> CheckpointManifest:
    manifest_path = checkpoint_manifest_path(iteration, save_folder)
    if not manifest_path.is_file():
        raise ValueError(f'Checkpoint manifest does not exist: {manifest_path}')
    manifest = CheckpointManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    if manifest.iteration != iteration:
        raise ValueError(f'Checkpoint manifest iteration {manifest.iteration} does not match {iteration}.')

    root = Path(save_folder)
    artifacts = (
        (root / manifest.model_path, manifest.model_sha256),
        (root / manifest.optimizer_path, manifest.optimizer_sha256),
        (root / manifest.jit_model_path, manifest.jit_model_sha256),
    )
    for artifact_path, expected_sha256 in artifacts:
        if not artifact_path.is_file():
            raise ValueError(f'Checkpoint artifact does not exist: {artifact_path}')
        if _sha256(artifact_path) != expected_sha256:
            raise ValueError(f'Checkpoint artifact hash does not match: {artifact_path}')

    for replay_reference in manifest.replay_files:
        replay_path = root / replay_reference.path
        if not replay_path.is_file():
            raise ValueError(f'Checkpoint replay file does not exist: {replay_path}')
        if replay_path.stat().st_size != replay_reference.size_bytes:
            raise ValueError(f'Checkpoint replay file size does not match: {replay_path}')
    return manifest


def create_model(args: NetworkParams, device: torch.device) -> Network:
    model = Network(args, device)
    # Not possible with JIT and model fusion model = try_compile(model)
    return model


def create_optimizer(model: Network, type: OptimizerType) -> torch.optim.Optimizer:
    if type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001, nesterov=True)
    elif type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True, eps=1e-5)
    raise ValueError(f'Optimizer type {type} not supported. Supported types: adamw, sgd')


def load_model(path: str | PathLike, args: NetworkParams, device: torch.device) -> Network:
    model = create_model(args, device)
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
    path: str | PathLike,
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
    iteration: int, args: NetworkParams, device: torch.device, save_folder: str | PathLike, type: OptimizerType
) -> tuple[Network, torch.optim.Optimizer]:
    manifest_path = checkpoint_manifest_path(iteration, save_folder)
    if iteration == 0 and not manifest_path.exists():
        model = create_model(args, device)
        optimizer = create_optimizer(model, type)
        log('Created a new model and optimizer for iteration 0.')
        return model, optimizer
    if iteration < 0:
        raise ValueError(f'Checkpoint iteration cannot be negative: {iteration}')

    manifest = load_checkpoint_manifest(iteration, save_folder)
    model = load_model(Path(save_folder) / manifest.model_path, args, device)
    optimizer = load_optimizer(Path(save_folder) / manifest.optimizer_path, model, type, device)
    log(f'Model and optimizer loaded from iteration {iteration}')
    return model, optimizer


def save_model_and_optimizer(
    model: Network, optimizer: torch.optim.Optimizer, iteration: int, save_folder: str | PathLike
) -> None:
    from src.settings import TRAINING_ARGS

    raw_model_path = model_save_path(iteration, save_folder)
    raw_optimizer_path = optimizer_save_path(iteration, save_folder)
    jit_model_path = raw_model_path.with_suffix('.jit.pt')

    temporary_model_path = _temporary_path(raw_model_path)
    temporary_optimizer_path = _temporary_path(raw_optimizer_path)
    temporary_jit_path = _temporary_path(jit_model_path)

    torch.save(model.state_dict(), temporary_model_path)
    torch.save(optimizer.state_dict(), temporary_optimizer_path)

    # Create a copy of the model, then set that to eval mode, fuse it, and save it as a JIT script
    fused_model = Network(TRAINING_ARGS.network, model.device)
    fused_model.load_state_dict(model.state_dict())
    fused_model.eval()
    fused_model.fuse_model()

    torch.jit.script(fused_model).save(str(temporary_jit_path))

    temporary_model_path.replace(raw_model_path)
    temporary_optimizer_path.replace(raw_optimizer_path)
    temporary_jit_path.replace(jit_model_path)

    manifest = CheckpointManifest(
        iteration=iteration,
        model_path=raw_model_path.name,
        model_sha256=_sha256(raw_model_path),
        optimizer_path=raw_optimizer_path.name,
        optimizer_sha256=_sha256(raw_optimizer_path),
        jit_model_path=jit_model_path.name,
        jit_model_sha256=_sha256(jit_model_path),
        replay_files=_replay_file_references(save_folder),
    )
    manifest_path = checkpoint_manifest_path(iteration, save_folder)
    temporary_manifest_path = _temporary_path(manifest_path)
    temporary_manifest_path.write_text(manifest.model_dump_json(indent=2) + '\n', encoding='utf-8')
    temporary_manifest_path.replace(manifest_path)


def get_latest_model_iteration(save_folder: str | PathLike) -> int:
    from src.settings import TRAINING_ARGS

    max_iteration = TRAINING_ARGS.num_iterations
    while max_iteration >= 0 and not checkpoint_manifest_path(max_iteration, save_folder).exists():
        max_iteration -= 1
    if max_iteration >= 0:
        load_checkpoint_manifest(max_iteration, save_folder)
        return max_iteration
    return 0
