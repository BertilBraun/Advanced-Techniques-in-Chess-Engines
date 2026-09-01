from __future__ import annotations

import math
from dataclasses import dataclass

from src.search_stopping.labels import CheckpointObservation
from src.search_stopping.targets import PolicyDistribution, policy_entropy, policy_kl, top_visit_share

# The stop-predictor input contract (plan section 4.2). Order is a binding contract with the
# native feature builder; native tests pin golden values against this implementation.
STOP_PREDICTOR_FEATURE_NAMES = (
    'top_share',
    'entropy',
    'top_gap',
    'kl_to_prior',
    'movement_kl',
    'segment_top_share',
    'root_value',
    'value_trend',
    'value_minus_network',
    'prior_top_share',
    'prior_entropy',
    'legal_move_count',
    'ply',
    'baseline_visits',
    'model_generation',
    'checkpoint_multiple',
    'root_warmth',
    'support_count',
    'top3_share',
)
STOP_PREDICTOR_FEATURE_COUNT = len(STOP_PREDICTOR_FEATURE_NAMES)


@dataclass(frozen=True)
class CheckpointFeatureContext:
    prior: PolicyDistribution
    network_root_value: float
    ply: int
    baseline_visits: int
    model_generation: int
    starting_visits: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.network_root_value):
            raise ValueError('The network root value must be finite.')
        if self.starting_visits < 0:
            raise ValueError('Checkpoint feature context requires nonnegative starting visits.')
        if self.ply < 0 or self.baseline_visits <= 0 or self.model_generation < 0:
            raise ValueError(
                'Checkpoint feature context requires a nonnegative ply and model generation and positive '
                'baseline visits.'
            )


def _top_two_gap(policy: PolicyDistribution) -> float:
    ordered = sorted(policy.probabilities, reverse=True)
    return ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)


def _third_share(policy: PolicyDistribution) -> float:
    ordered = sorted(policy.probabilities, reverse=True)
    return ordered[2] if len(ordered) > 2 else 0.0


def _segment_distribution(
    current: CheckpointObservation,
    previous: CheckpointObservation | None,
    prior: PolicyDistribution,
) -> PolicyDistribution:
    if previous is None:
        return current.policy
    weights = []
    for current_probability, previous_probability in zip(
        current.policy.probabilities, previous.policy.probabilities, strict=True
    ):
        weights.append(max(0.0, current.visits * current_probability - previous.visits * previous_probability))
    total = sum(weights)
    if total <= 0.0:
        return current.policy
    return PolicyDistribution(probabilities=tuple(weight / total for weight in weights))


def checkpoint_feature_vector(
    current: CheckpointObservation,
    previous: CheckpointObservation | None,
    context: CheckpointFeatureContext,
    checkpoint_multiple: float,
) -> tuple[float, ...]:
    """`previous` is the prior checkpoint, or the zeroth checkpoint at `starting_visits` for i=1;
    `None` means a genuinely fresh root, where the raw prior and network value stand in for it."""
    previous_policy = previous.policy if previous is not None else context.prior
    previous_value = previous.root_value if previous is not None else context.network_root_value
    segment = _segment_distribution(current, previous, context.prior)
    legal_move_count = sum(probability > 0.0 for probability in context.prior.probabilities)
    vector = (
        top_visit_share(current.policy),
        policy_entropy(current.policy),
        _top_two_gap(current.policy),
        policy_kl(current.policy, context.prior),
        policy_kl(current.policy, previous_policy),
        top_visit_share(segment),
        current.root_value,
        current.root_value - previous_value,
        current.root_value - context.network_root_value,
        top_visit_share(context.prior),
        policy_entropy(context.prior),
        float(legal_move_count),
        float(context.ply),
        float(context.baseline_visits),
        float(context.model_generation),
        checkpoint_multiple,
        context.starting_visits / context.baseline_visits,
        float(sum(probability > 0.0 for probability in current.policy.probabilities)),
        _third_share(current.policy),
    )
    if any(not math.isfinite(value) for value in vector):
        raise ValueError('Stop-predictor features must be finite.')
    return vector
