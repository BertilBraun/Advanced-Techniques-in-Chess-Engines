from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from src.games.contracts import WdlTarget
from src.packed_planes import PackedPlanePayload
from src.self_play.completed_game import SparseSearchVisit


@dataclass(frozen=True)
class SparsePolicyTarget:
    visits: tuple[SparseSearchVisit, ...]

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


AuxiliaryReplayTarget: TypeAlias = EligibleNextPolicyTarget | IneligibleNextPolicyTarget


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
