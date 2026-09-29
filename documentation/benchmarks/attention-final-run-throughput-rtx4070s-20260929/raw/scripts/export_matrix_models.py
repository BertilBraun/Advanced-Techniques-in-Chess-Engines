"""Exports random-initialised inference models for the throughput matrix; throughput depends only on the architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.checkpoint.persistence import create_model
from src.training.network import InferenceNetwork

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
attention = load_experiment_configuration(Path('configs/production/vast-chess-8gpu-final-attention.yaml'))
final = load_experiment_configuration(Path('configs/production/chess-final-config.yaml'))
attention_models = {model.model_id: model.network for model in attention.training.progressive_model_sizing.models}
small = attention_models['chess-lc0-attention-8x160']
medium = attention_models['chess-lc0-attention-10x192']


def widened(layers: int, embedding: int):
    return medium.model_copy(
        update={
            'num_layers': layers,
            'embedding_size': embedding,
            'num_heads': embedding // 32,
            'feedforward_size': 4 * embedding,
            'smolgen': medium.smolgen.model_copy(update={'hidden_size': embedding, 'generated_size': embedding}),
            'policy_head': medium.policy_head.model_copy(update={'key_size': embedding}),
        }
    )


architectures = {
    'attention-8x160': small,
    'attention-10x192': medium,
    'attention-12x192': widened(12, 192),
    'attention-10x224': widened(10, 224),
    'cnn-14x160': next(
        model.network
        for model in final.training.progressive_model_sizing.models
        if model.model_id == 'chess-cnn-scaled-post-14x160-fromto-int8'
    ),
}
assert architectures['attention-12x192'].model_dump() != medium.model_dump()
for name, architecture in architectures.items():
    torch.manual_seed(20260929)
    model = create_model(architecture, torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    inference = InferenceNetwork(model)
    inference.eval()
    inference.fuse_model()
    path = output / f'{name}.pt'
    torch.jit.save(
        torch.jit.script(inference),
        str(path),
        _extra_files={'network.json': inference.checkpoint_definition().model_dump_json()},
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f'{name}: {parameters:,} parameters -> {path}')
