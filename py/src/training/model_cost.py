from __future__ import annotations

from dataclasses import dataclass

import torch
from src.training.network import Network
from torch.utils.flop_counter import FlopCounterMode


@dataclass(frozen=True)
class ParameterCost:
    trunk: int
    policy_head: int
    value_head: int
    auxiliary_heads: int

    @property
    def total(self) -> int:
        return self.trunk + self.policy_head + self.value_head + self.auxiliary_heads


@dataclass(frozen=True)
class MultiplyAccumulateCost:
    trunk: int
    policy_head: int
    value_head: int

    @property
    def total(self) -> int:
        return self.trunk + self.policy_head + self.value_head


@dataclass(frozen=True)
class ModelCost:
    parameters: ParameterCost
    multiply_accumulates_per_position: MultiplyAccumulateCost

    @property
    def multiply_accumulates_per_parameter(self) -> float:
        return self.multiply_accumulates_per_position.total / self.parameters.total


def _count_parameters(*modules: torch.nn.Module) -> int:
    seen: dict[int, int] = {}
    for module in modules:
        for parameter in module.parameters():
            seen[id(parameter)] = parameter.numel()
    return sum(seen.values())


def _count_multiply_accumulates(callable_under_test, batch_size: int) -> int:
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        callable_under_test()
    # FlopCounterMode reports two floating-point operations per multiply-accumulate.
    return counter.get_total_flops() // (2 * batch_size)


def measure_model_cost(model: Network, batch_size: int = 8) -> ModelCost:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dimensions = model.dimensions
    states = torch.zeros(
        (batch_size, dimensions.channels, dimensions.rows, dimensions.columns),
        device=device,
    )
    with torch.no_grad():
        features = model.trunk_features(states)

    cost = ModelCost(
        parameters=ParameterCost(
            trunk=_count_parameters(model.start_block, model.backbone, model.finish_block),
            policy_head=_count_parameters(model.policy_head),
            value_head=_count_parameters(model.value_head),
            auxiliary_heads=_count_parameters(model.auxiliary_head_modules),
        ),
        multiply_accumulates_per_position=MultiplyAccumulateCost(
            trunk=_count_multiply_accumulates(lambda: model.trunk_features(states), batch_size),
            policy_head=_count_multiply_accumulates(lambda: model.policy_head(features), batch_size),
            value_head=_count_multiply_accumulates(lambda: model.value_head(features), batch_size),
        ),
    )
    model.train(was_training)
    return cost


def format_model_cost(name: str, cost: ModelCost) -> str:
    parameters = cost.parameters
    multiply_accumulates = cost.multiply_accumulates_per_position
    return (
        f'{name}: {parameters.total:,} parameters = {parameters.trunk:,} trunk + '
        f'{parameters.policy_head:,} policy head + {parameters.value_head:,} value head + '
        f'{parameters.auxiliary_heads:,} auxiliary heads; '
        f'{multiply_accumulates.total:,} MAC per position = {multiply_accumulates.trunk:,} trunk + '
        f'{multiply_accumulates.policy_head:,} policy head + {multiply_accumulates.value_head:,} value head; '
        f'{cost.multiply_accumulates_per_parameter:.1f} MAC per parameter.'
    )
