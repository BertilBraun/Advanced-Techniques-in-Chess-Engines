"""Adapts a scripted Lc0 teacher to the surface the distillation dataset builder expects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TeacherOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    auxiliary_logits: tuple[torch.Tensor, ...]


class Lc0TeacherNetwork:
    """Presents the scripted teacher as a project network.

    The scripted module already returns WDL as probabilities, so they are handed back as logarithms:
    the caller softmaxes them, and the softmax of a log-distribution is that distribution again.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def training_output(self, encoded: torch.Tensor) -> TeacherOutput:
        policy_logits, wdl_probabilities = self._module(encoded)
        return TeacherOutput(
            policy_logits=policy_logits,
            wdl_logits=torch.log(wdl_probabilities.clamp_min(1e-9)),
            auxiliary_logits=(),
        )


@dataclass(frozen=True)
class Lc0Teacher:
    network: Lc0TeacherNetwork
    parameter_count: int


def load_lc0_teacher(path: Path, device: torch.device) -> Lc0Teacher:
    module = torch.jit.load(str(path), map_location=device).eval()
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    return Lc0Teacher(network=Lc0TeacherNetwork(module), parameter_count=parameter_count)


def sampling_temperature_at(
    ply: int,
    starting_temperature: float,
    final_temperature: float,
    greedy_after_ply: int | None,
) -> float:
    """Production interpolates temperature across the greedy ply; a fixed value is the older behaviour."""
    if greedy_after_ply is None:
        return starting_temperature
    progress = min(ply / greedy_after_ply, 1.0)
    return starting_temperature + (final_temperature - starting_temperature) * progress
