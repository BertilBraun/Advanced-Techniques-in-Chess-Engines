from __future__ import annotations

from os import PathLike
from pathlib import Path


def model_save_path(generation: int, save_folder: str | PathLike[str]) -> Path:
    path = Path(save_folder) / f'model_{generation}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def optimizer_save_path(generation: int, save_folder: str | PathLike[str]) -> Path:
    path = Path(save_folder) / f'optimizer_{generation}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_manifest_path(generation: int, save_folder: str | PathLike[str]) -> Path:
    return Path(save_folder) / f'checkpoint_{generation}.json'


def inference_model_path(path: str | PathLike[str]) -> Path:
    model_path = Path(path)
    if model_path.name.endswith('.jit.pt'):
        return model_path
    if model_path.suffix == '.pt':
        return model_path.with_suffix('.jit.pt')
    raise ValueError(f'Model path must end in .pt or .jit.pt: {model_path}')
