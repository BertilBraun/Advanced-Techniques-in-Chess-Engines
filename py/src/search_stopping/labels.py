from __future__ import annotations

import math
from dataclasses import dataclass

from src.search_stopping.targets import PolicyDistribution, policy_kl


@dataclass(frozen=True)
class CheckpointObservation:
    visits: int
    root_value: float
    policy: PolicyDistribution


@dataclass(frozen=True)
class CappedSearchRecord:
    checkpoints: tuple[CheckpointObservation, ...]
    final_visits: int
    final_root_value: float
    final_policy: PolicyDistribution

    def __post_init__(self) -> None:
        visits = tuple(checkpoint.visits for checkpoint in self.checkpoints)
        if visits != tuple(sorted(set(visits))):
            raise ValueError('Capped-search checkpoints must be unique and increasing.')
        if visits and visits[-1] >= self.final_visits:
            raise ValueError('Capped-search checkpoints must lie strictly below the final visit count.')


@dataclass(frozen=True)
class CheckpointStopLabel:
    kl_to_final: float
    value_gap: float
    uncertain: bool
    argmax_swap: bool


def checkpoint_stop_labels(
    record: CappedSearchRecord,
    eps_pi: float,
    eps_v: float,
) -> tuple[CheckpointStopLabel, ...]:
    """Section 3.1 of the adaptive-stopping plan: uncertain at c_i iff any remaining checkpoint
    (the future-max clause) diverges from the cap distribution by at least eps in policy or value."""
    if not math.isfinite(eps_pi) or eps_pi <= 0.0 or not math.isfinite(eps_v) or eps_v <= 0.0:
        raise ValueError('Label epsilons must be finite and positive.')
    divergences = tuple(policy_kl(record.final_policy, checkpoint.policy) for checkpoint in record.checkpoints)
    value_gaps = tuple(abs(checkpoint.root_value - record.final_root_value) for checkpoint in record.checkpoints)
    if any(not math.isfinite(value) for value in (*divergences, *value_gaps)):
        raise ValueError('Checkpoint label reconstruction must be finite.')
    final_argmax = _argmax(record.final_policy)
    labels = []
    count = len(record.checkpoints)
    for index in range(count):
        future_kl = max(divergences[index:])
        future_gap = max(value_gaps[index:])
        labels.append(
            CheckpointStopLabel(
                kl_to_final=divergences[index],
                value_gap=value_gaps[index],
                uncertain=future_kl >= eps_pi or future_gap >= eps_v,
                argmax_swap=_argmax(record.checkpoints[index].policy) != final_argmax,
            )
        )
    return tuple(labels)


def _argmax(policy: PolicyDistribution) -> int:
    return max(range(len(policy.probabilities)), key=policy.probabilities.__getitem__)
