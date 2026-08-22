from __future__ import annotations

import torch

from src.training.batch import TrainingBatch


def training_batch(
    *,
    policy_targets: torch.Tensor,
    wdl_targets: torch.Tensor,
    root_values: torch.Tensor | None = None,
    policy_legal_action_ids: torch.Tensor | None = None,
    auxiliary_targets: tuple[torch.Tensor, ...] = (),
    auxiliary_legal_action_ids: tuple[torch.Tensor, ...] = (),
    auxiliary_eligibility: tuple[torch.Tensor, ...] = (),
) -> TrainingBatch:
    batch_size = policy_targets.shape[0]
    return TrainingBatch(
        states=torch.zeros((batch_size, 1)),
        policy_targets=policy_targets,
        policy_legal_action_ids=(
            torch.tensor(((0, 1),) * batch_size) if policy_legal_action_ids is None else policy_legal_action_ids
        ),
        wdl_targets=wdl_targets,
        root_values=torch.zeros(batch_size) if root_values is None else root_values,
        auxiliary_targets=auxiliary_targets,
        auxiliary_legal_action_ids=auxiliary_legal_action_ids,
        auxiliary_eligibility=auxiliary_eligibility,
        sample_weights=torch.ones(batch_size),
        source_model_generations=torch.zeros(batch_size, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(batch_size, dtype=torch.float64),
    )
