from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic, Protocol, TypeVar


PositionT = TypeVar('PositionT')


@dataclass(frozen=True)
class EnginePolicyEntry:
    action_id: int
    probability: float

    def __post_init__(self) -> None:
        if self.action_id < 0:
            raise ValueError('Engine policy action IDs must be nonnegative.')
        if not isfinite(self.probability) or self.probability <= 0.0:
            raise ValueError('Engine policy probabilities must be finite and positive.')


@dataclass(frozen=True)
class EnginePolicy:
    entries: tuple[EnginePolicyEntry, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError('Engine policy must contain at least one action.')
        action_ids = tuple(entry.action_id for entry in self.entries)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError('Engine policy action IDs must be unique.')
        if abs(sum(entry.probability for entry in self.entries) - 1.0) > 1e-6:
            raise ValueError('Engine policy probabilities must sum to one.')

    @property
    def top_action_id(self) -> int:
        return min(
            self.entries,
            key=lambda entry: (-entry.probability, entry.action_id),
        ).action_id


class EnginePolicyProvider(Protocol, Generic[PositionT]):
    @property
    def game_name(self) -> str: ...

    @property
    def rules_digest(self) -> str: ...

    @property
    def representation_digest(self) -> str: ...

    @property
    def engine_identity(self) -> str: ...

    @property
    def engine_artifact_sha256(self) -> tuple[str, ...]: ...

    @property
    def label_search_limit(self) -> int: ...

    def policy(self, position: PositionT, action_ids: tuple[int, ...]) -> EnginePolicy: ...

    def render_game(self, action_ids: tuple[int, ...]) -> str: ...

    def close(self) -> None: ...


def validate_engine_policy(policy: EnginePolicy, legal_actions: tuple[int, ...]) -> EnginePolicy:
    legal = set(legal_actions)
    if any(entry.action_id not in legal for entry in policy.entries):
        raise ValueError('Engine policy contains an illegal action.')
    return policy
