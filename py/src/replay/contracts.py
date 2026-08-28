from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

from src.games.contracts import WdlTarget
from src.games.representation import PackedPlanePayload
from src.self_play.completed_game import SearchVisitCounts


@dataclass(frozen=True)
class SparsePolicyTarget:
    visits: SearchVisitCounts
    legal_action_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        action_ids = self.visits.action_ids
        if not self.legal_action_ids or any(action_id < 0 for action_id in self.legal_action_ids):
            raise ValueError('Sparse policy targets require nonnegative legal action IDs.')
        if len(set(self.legal_action_ids)) != len(self.legal_action_ids):
            raise ValueError('Sparse policy legal action IDs must be unique.')
        if not set(action_ids).issubset(self.legal_action_ids):
            raise ValueError('Sparse policy visits must be legal.')


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


@dataclass(frozen=True)
class EligibleScalarAuxiliaryTarget:
    kind: Literal['future_search_value', 'irreversible_progress']
    value: float
    eligible: Literal[True] = True


@dataclass(frozen=True)
class IneligibleScalarAuxiliaryTarget:
    kind: Literal['future_search_value', 'irreversible_progress']
    eligible: Literal[False] = False


@dataclass(frozen=True)
class EligibleSearchBudgetTarget:
    normalized_target: float
    raw_kl: float
    prediction_logit: float
    predicted_quantile: float
    source_generation: int
    model_generation: int
    inference_model_sha256: str
    kind: Literal['search_budget'] = 'search_budget'
    eligible: Literal[True] = True

    def __post_init__(self) -> None:
        if not isfinite(self.normalized_target) or not 0.0 <= self.normalized_target <= 1.0:
            raise ValueError('Search-budget targets must lie in [0, 1].')
        if not isfinite(self.raw_kl) or self.raw_kl < 0.0:
            raise ValueError('Search-budget raw KL must be finite and nonnegative.')
        if not isfinite(self.prediction_logit):
            raise ValueError('Search-budget prediction logits must be finite.')
        if not isfinite(self.predicted_quantile) or not 0.0 <= self.predicted_quantile <= 1.0:
            raise ValueError('Search-budget predicted quantiles must lie in [0, 1].')
        if self.source_generation < 0 or self.model_generation < 0:
            raise ValueError('Search-budget source and model generations must be nonnegative.')
        if len(self.inference_model_sha256) != 64 or any(
            character not in '0123456789abcdef' for character in self.inference_model_sha256
        ):
            raise ValueError('Search-budget model lineage must be a lowercase SHA-256 digest.')


@dataclass(frozen=True)
class IneligibleSearchBudgetTarget:
    kind: Literal['search_budget'] = 'search_budget'
    eligible: Literal[False] = False


@dataclass(frozen=True)
class EligibleLegalMovesTarget:
    kind: Literal['legal_moves'] = 'legal_moves'
    eligible: Literal[True] = True


AuxiliaryReplayTarget: TypeAlias = (
    EligibleNextPolicyTarget
    | IneligibleNextPolicyTarget
    | EligibleRemainingGameLengthTarget
    | IneligibleRemainingGameLengthTarget
    | EligibleScalarAuxiliaryTarget
    | IneligibleScalarAuxiliaryTarget
    | EligibleSearchBudgetTarget
    | IneligibleSearchBudgetTarget
    | EligibleLegalMovesTarget
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
