from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from src.games.representation import NetworkDimensions
from src.training.checkpoint.persistence import create_model
from src.training.network import Network, NetworkConfiguration, NetworkDefinition
from src.training.targets import AuxiliaryHeadLayout


@dataclass(frozen=True)
class LoadedTeacher:
    network: Network
    definition: NetworkDefinition
    generation: int

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # torch.compile prefixes every parameter of the wrapped module.
    return {key.removeprefix('_orig_mod.'): tensor for key, tensor in state_dict.items()}


def read_network_definition(checkpoint_manifest_path: Path) -> NetworkDefinition | None:
    manifest = json.loads(checkpoint_manifest_path.read_text(encoding='utf-8'))
    if 'network' not in manifest:
        return None
    return NetworkDefinition.model_validate(manifest['network'])


def load_teacher(
    weights_path: Path,
    architecture: NetworkConfiguration,
    dimensions: NetworkDimensions,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...],
    device: torch.device,
    generation: int,
) -> LoadedTeacher:
    network = create_model(architecture, device, dimensions, auxiliary_heads)
    state_dict = normalize_state_dict_keys(torch.load(weights_path, map_location=device, weights_only=True))
    expected = set(network.state_dict())
    # Heads the caller did not ask for stay in the file; every head it did ask for must be present.
    network.load_state_dict({key: tensor for key, tensor in state_dict.items() if key in expected})
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    definition = NetworkDefinition(architecture=architecture, dimensions=dimensions, auxiliary_heads=auxiliary_heads)
    return LoadedTeacher(network=network, definition=definition, generation=generation)
