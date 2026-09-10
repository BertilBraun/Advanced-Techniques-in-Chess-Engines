from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from src.evaluation.inference import PolicyActionSelector
from src.games.contracts import GameStateContract


class _FakeState:
    def legal_action_ids(self, position: int) -> tuple[int, ...]:
        return (position % 3, 3)


class _FakeModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.batch_sizes.append(len(states))
        policy = torch.arange(4, dtype=torch.float32).repeat(len(states), 1)
        return policy, torch.zeros((len(states), 3))


def test_policy_selector_caps_direct_inference_batches_and_masks_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decoded_inputs(state: GameStateContract[int], positions: tuple[int, ...]) -> np.ndarray:
        return np.zeros((len(positions), 1), dtype=np.float32)

    monkeypatch.setattr(
        'src.evaluation.inference.decode_network_inputs',
        decoded_inputs,
    )
    selector = PolicyActionSelector.__new__(PolicyActionSelector)
    selector.state = cast(GameStateContract[int], _FakeState())
    selector.device = torch.device('cpu')
    model = _FakeModel()
    selector.model = cast(torch.jit.ScriptModule, model)
    selector.maximum_batch_size = 2

    selected = selector.choose_actions((0, 1, 2, 3, 4))

    assert model.batch_sizes == [2, 2, 1]
    assert selected == (3, 3, 3, 3, 3)
