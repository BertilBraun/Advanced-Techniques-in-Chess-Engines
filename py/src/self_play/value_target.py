from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite


REPLAY_SCHEMA_VERSION = 3


class FinalOutcome(IntEnum):
    """Stable network WDL order shared by replay, Torch, JIT, and C++."""

    WIN = 0
    DRAW = 1
    LOSS = 2

    @classmethod
    def from_score(cls, score: float) -> FinalOutcome:
        if not isfinite(score) or score < -1.0 or score > 1.0:
            raise ValueError(f'Outcome score must be finite and in [-1, 1], got {score}.')
        if score > 0.0:
            return cls.WIN
        if score < 0.0:
            return cls.LOSS
        return cls.DRAW

    @property
    def expected_score(self) -> float:
        if self is FinalOutcome.WIN:
            return 1.0
        if self is FinalOutcome.LOSS:
            return -1.0
        return 0.0


class TerminationReason(IntEnum):
    NATURAL = 0
    RESIGNATION = 1
    PLY_CAP = 2
    MATERIAL_ADJUDICATION = 3
    DIAGNOSTIC = 4

    @property
    def permits_outcome_target(self) -> bool:
        return self in (TerminationReason.NATURAL, TerminationReason.RESIGNATION)

    @property
    def permits_mcts_target(self) -> bool:
        return self is not TerminationReason.DIAGNOSTIC


@dataclass(frozen=True)
class ReplayValueTarget:
    final_outcome: FinalOutcome
    mcts_root_value: float
    termination_reason: TerminationReason
    outcome_target_eligible: bool

    def __post_init__(self) -> None:
        if not isfinite(self.mcts_root_value) or not -1.0 <= self.mcts_root_value <= 1.0:
            raise ValueError(f'MCTS root value must be finite and in [-1, 1], got {self.mcts_root_value}.')
        if self.outcome_target_eligible != self.termination_reason.permits_outcome_target:
            raise ValueError(
                f'Outcome eligibility {self.outcome_target_eligible} conflicts with '
                f'termination reason {self.termination_reason.name}.'
            )

    @classmethod
    def from_scores(
        cls,
        final_score: float,
        mcts_root_value: float,
        termination_reason: TerminationReason,
    ) -> ReplayValueTarget:
        return cls(
            final_outcome=FinalOutcome.from_score(final_score),
            mcts_root_value=mcts_root_value,
            termination_reason=termination_reason,
            outcome_target_eligible=termination_reason.permits_outcome_target,
        )


def outcome_from_sample_perspective(
    game_outcome: float,
    final_current_player: int,
    sample_current_player: int,
) -> float:
    if final_current_player not in (-1, 1) or sample_current_player not in (-1, 1):
        raise ValueError('Player perspectives must be -1 or 1.')
    return game_outcome if sample_current_player == final_current_player else -game_outcome
