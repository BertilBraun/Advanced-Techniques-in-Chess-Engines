from __future__ import annotations

import sys

import torch

sys.path.insert(0, '.')

from src.games.representation import NetworkDimensions
from src.training.network import (
    AttentionNetworkParams,
    Chess76PlaneDirectPolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    Network,
    NetworkParams,
    ResidualContextPlacement,
)

DIMENSIONS = NetworkDimensions(channels=29, rows=8, columns=8, actions=4864)

torch.manual_seed(0)
probe_input = (torch.rand(64, DIMENSIONS.channels, 8, 8) < 0.15).float()
probe_input[:, 22:] = torch.rand(64, 7, 1, 1)


def probe(arguments: AttentionNetworkParams | NetworkParams, name: str, train_mode: bool = True) -> None:
    network = Network(arguments, torch.device('cpu'), DIMENSIONS)
    network.train(train_mode)
    with torch.no_grad():
        features = network._features(probe_input)
        policy_logits, value_logits = network.logit_forward(probe_input)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    cross_entropy = torch.nn.functional.cross_entropy(policy_logits, torch.randint(0, 4864, (64,)))
    print(
        f'{name}: params={parameter_count / 1e6:.2f}M'
        f' feat_rms={features.pow(2).mean().sqrt():.1f}'
        f' policy_logit_std={policy_logits.std():.1f}'
        f' max|logit|={policy_logits.abs().max():.0f}'
        f' CE(rand target)={cross_entropy:.1f}'
        f' value_logit_std={value_logits.std():.2f}'
    )


policy_head = Chess76PlaneDirectPolicyHeadConfiguration()
for layers, embedding, heads, feedforward, name in (
    (6, 96, 3, 192, 'att 6x96 pilot'),
    (8, 128, 4, 256, 'att 8x128 1m'),
    (10, 160, 5, 320, 'att 10x160 2m'),
    (15, 192, 6, 384, 'att 15x192 4m5'),
):
    probe(
        AttentionNetworkParams(
            policy_head=policy_head,
            num_layers=layers,
            embedding_size=embedding,
            num_heads=heads,
            feedforward_size=feedforward,
        ),
        name,
    )
for hidden_size, name in ((144, 'cnn 12x144 76plane'), (112, 'cnn 12x112 76plane')):
    probe(
        NetworkParams(
            policy_head=policy_head,
            num_layers=12,
            hidden_size=hidden_size,
            residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        ),
        name,
    )
