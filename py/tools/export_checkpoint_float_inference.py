from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from src.training.checkpoint.contracts import load_checkpoint_manifest_path
from src.training.network import InferenceNetwork, Network
from src.util.hashing import file_sha256
from torch import Tensor


@dataclass(frozen=True)
class ExportResult:
    checkpoint_manifest: str
    checkpoint_generation: int
    output: str
    output_sha256: str


def export_float_inference(checkpoint_manifest_path: Path, generation: int, output_path: Path) -> ExportResult:
    manifest = load_checkpoint_manifest_path(checkpoint_manifest_path, generation)
    state_dict: dict[str, Tensor] = torch.load(
        checkpoint_manifest_path.parent / manifest.model_path,
        map_location='cpu',
        weights_only=True,
    )
    model = Network(
        manifest.network.architecture,
        torch.device('cpu'),
        manifest.network.dimensions,
        manifest.network.auxiliary_heads,
    )
    model.load_state_dict({name: state_dict[name] for name in model.state_dict()})
    inference_model = InferenceNetwork(model)
    inference_model.eval()
    inference_model.fuse_model()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        torch.jit.script(inference_model),
        str(output_path),
        _extra_files={'network.json': inference_model.checkpoint_definition().model_dump_json()},
    )
    return ExportResult(
        checkpoint_manifest=str(checkpoint_manifest_path),
        checkpoint_generation=generation,
        output=str(output_path),
        output_sha256=file_sha256(output_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Export a floating-point TorchScript inference model from a checkpoint.'
    )
    parser.add_argument('--checkpoint-manifest', required=True, type=Path)
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.generation < 0:
        raise ValueError('Generation must be nonnegative.')
    print(
        json.dumps(
            asdict(export_float_inference(arguments.checkpoint_manifest, arguments.generation, arguments.output))
        )
    )


if __name__ == '__main__':
    main()
