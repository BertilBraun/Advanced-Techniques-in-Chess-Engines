from __future__ import annotations

from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np
import torch

from src.games.contracts import GameStateContract
from src.games.representation import PackedPlanePayload, decode_packed_planes_into


PositionT = TypeVar('PositionT')


def decode_network_inputs(
    state: GameStateContract[PositionT],
    positions: tuple[PositionT, ...],
) -> np.ndarray:
    return decode_packed_inputs(state, tuple(state.encode_network_input(position) for position in positions))


def decode_packed_inputs(
    state: GameStateContract[PositionT],
    packed_inputs: tuple[PackedPlanePayload, ...],
) -> np.ndarray:
    decoded = np.empty(
        (
            len(packed_inputs),
            state.representation.channels,
            state.representation.rows,
            state.representation.columns,
        ),
        dtype=np.float32,
    )
    decode_packed_planes_into(
        packed_inputs,
        state.packed_plane_layout,
        state.representation.binary_channels,
        state.representation.scalar_channels,
        decoded,
    )
    return decoded


class PolicyActionSelector(Generic[PositionT]):
    def __init__(
        self,
        state: GameStateContract[PositionT],
        model_path: Path,
        device_id: int,
        device_type: Literal['cpu', 'cuda'],
    ) -> None:
        self.state = state
        self.device = torch.device('cpu') if device_type == 'cpu' else torch.device('cuda', device_id)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def choose_actions(self, positions: tuple[PositionT, ...]) -> tuple[int, ...]:
        if not positions:
            return ()
        decoded = decode_network_inputs(self.state, positions)
        with torch.inference_mode():
            policy_logits, _ = self.model(torch.from_numpy(decoded).to(self.device))
        policy_logits = policy_logits.float().cpu()
        return tuple(
            max(
                self.state.legal_action_ids(position),
                key=lambda action_id: (float(policy_logits[row_index, action_id]), -action_id),
            )
            for row_index, position in enumerate(positions)
        )
