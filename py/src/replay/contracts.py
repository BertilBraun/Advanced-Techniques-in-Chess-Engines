from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from AlphaZeroCpp import GameSearchVisit

from src.games.contracts import WdlTarget
from src.games.representation import PackedPlanePayload


@dataclass(frozen=True)
class SparsePolicyTarget:
    visits: tuple[GameSearchVisit, ...]

    def __post_init__(self) -> None:
        if not self.visits or any(visit.visit_count <= 0 for visit in self.visits):
            raise ValueError('Sparse policy targets require positive visits.')
        action_ids = tuple(visit.action_id for visit in self.visits)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError('Sparse policy action IDs must be unique.')


@dataclass(frozen=True)
class EligibleNextPolicyTarget:
    policy: SparsePolicyTarget
    kind: Literal['next_policy'] = 'next_policy'
    eligible: Literal[True] = True


@dataclass(frozen=True)
class IneligibleNextPolicyTarget:
    kind: Literal['next_policy'] = 'next_policy'
    eligible: Literal[False] = False


@dataclass(frozen=True)
class EligibleRemainingGameLengthTarget:
    normalized_length: float
    kind: Literal['remaining_game_length'] = 'remaining_game_length'
    eligible: Literal[True] = True

    def __post_init__(self) -> None:
        if self.normalized_length < 0.0:
            raise ValueError('Normalized remaining game length must be nonnegative.')


@dataclass(frozen=True)
class IneligibleRemainingGameLengthTarget:
    kind: Literal['remaining_game_length'] = 'remaining_game_length'
    eligible: Literal[False] = False


AuxiliaryReplayTarget: TypeAlias = (
    EligibleNextPolicyTarget
    | IneligibleNextPolicyTarget
    | EligibleRemainingGameLengthTarget
    | IneligibleRemainingGameLengthTarget
)


@dataclass(frozen=True)
class ReplaySample:
    encoded_state: PackedPlanePayload
    policy: SparsePolicyTarget
    wdl_target: WdlTarget
    root_value: float
    auxiliary_targets: tuple[AuxiliaryReplayTarget, ...]
    sample_weight: float
    source_model_generation: int
    source_created_at_seconds: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.root_value <= 1.0:
            raise ValueError('Replay root value must lie in [-1, 1].')
        if self.sample_weight <= 0.0 or self.source_model_generation < 0 or self.source_created_at_seconds < 0.0:
            raise ValueError('Replay source metadata and sample weight must be nonnegative.')
