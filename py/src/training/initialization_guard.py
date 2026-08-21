from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from src.training.network import Network
from torch import Tensor

MAXIMUM_POLICY_LOGIT_STD = 1.5
MINIMUM_POLICY_ENTROPY_RATIO = 0.6


@dataclass(frozen=True)
class InitializationProbe:
    policy_logit_std: float
    policy_entropy_ratio: float


def probe_policy_initialization(model: Network, states: Tensor, legal_action_ids: Tensor) -> InitializationProbe:
    # Batch-normalization running statistics are untrained at generation 0, so probe the training forward.
    was_training = model.training
    model.train(True)
    try:
        with torch.no_grad():
            policy_logits, _ = model.logit_forward(states)
    finally:
        model.train(was_training)
    return summarize_policy_initialization(policy_logits, legal_action_ids)


def summarize_policy_initialization(policy_logits: Tensor, legal_action_ids: Tensor) -> InitializationProbe:
    logit_stds: list[float] = []
    entropy_ratios: list[float] = []
    for row_logits, row_legal_action_ids in zip(policy_logits, legal_action_ids):
        legal_ids = row_legal_action_ids[row_legal_action_ids >= 0]
        if legal_ids.numel() < 2:
            continue
        legal_logits = row_logits[legal_ids].double()
        logit_stds.append(float(legal_logits.std(correction=0)))
        log_probabilities = torch.log_softmax(legal_logits, dim=0)
        entropy = float(-(log_probabilities.exp() * log_probabilities).sum())
        entropy_ratios.append(entropy / math.log(legal_ids.numel()))
    if not logit_stds:
        raise ValueError('Initialization probe requires at least one position with two or more legal actions.')
    return InitializationProbe(
        policy_logit_std=sum(logit_stds) / len(logit_stds),
        policy_entropy_ratio=sum(entropy_ratios) / len(entropy_ratios),
    )


def assert_healthy_policy_initialization(probe: InitializationProbe) -> None:
    if probe.policy_logit_std > MAXIMUM_POLICY_LOGIT_STD:
        raise RuntimeError(
            f'Initial legal-masked policy-logit std {probe.policy_logit_std:.3f} exceeds '
            f'{MAXIMUM_POLICY_LOGIT_STD}; the policy head initialization is unhealthy.'
        )
    if probe.policy_entropy_ratio < MINIMUM_POLICY_ENTROPY_RATIO:
        raise RuntimeError(
            f'Initial legal-masked policy entropy ratio {probe.policy_entropy_ratio:.3f} is below '
            f'{MINIMUM_POLICY_ENTROPY_RATIO}; the policy head initialization is unhealthy.'
        )
