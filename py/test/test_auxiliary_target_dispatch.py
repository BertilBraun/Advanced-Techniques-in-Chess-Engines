from __future__ import annotations

from typing import get_args

import pytest
import torch
from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.distributions import capture_training_distributions
from src.training.objective import ResolvedTrainingObjective, resolve_auxiliary_losses
from src.training.targets import (
    AuxiliaryTargetConfiguration,
    FutureSearchValueTargetConfiguration,
    IrreversibleProgressTargetConfiguration,
    LegalMovesTargetConfiguration,
    NextPolicyTargetConfiguration,
    RemainingGameLengthTargetConfiguration,
    ValueResidualTargetConfiguration,
    build_training_target_layout,
)

ACTION_SIZE = 8

EVERY_TARGET: tuple[AuxiliaryTargetConfiguration, ...] = (
    NextPolicyTargetConfiguration(ply_offset=1, loss_weight=0.1),
    RemainingGameLengthTargetConfiguration(loss_weight=0.1, normalization_scale=400.0),
    FutureSearchValueTargetConfiguration(ply_offset=2, smooth_l1_beta=0.1, loss_weight=0.1),
    IrreversibleProgressTargetConfiguration(horizon_plies=20, loss_weight=0.1),
    ValueResidualTargetConfiguration(smooth_l1_beta=0.05, loss_weight=0.1),
    LegalMovesTargetConfiguration(loss_weight=0.1),
)


def test_every_configured_auxiliary_kind_is_exercised_here() -> None:
    # Without this the suite silently stops covering a kind the moment one is added, which is how
    # four separate dispatches reached a production run with no arm for a new target.
    union_kinds = {
        get_args(member.model_fields['kind'].annotation)[0]
        for member in get_args(get_args(AuxiliaryTargetConfiguration)[0])
    }

    assert {target.kind for target in EVERY_TARGET} == union_kinds


@pytest.mark.parametrize('target', EVERY_TARGET, ids=lambda target: target.kind)
def test_every_auxiliary_kind_resolves_a_loss(target: AuxiliaryTargetConfiguration) -> None:
    (resolved,) = resolve_auxiliary_losses((target,), model_generation=0)

    assert resolved.kind == target.kind


@pytest.mark.parametrize('target', EVERY_TARGET, ids=lambda target: target.kind)
def test_every_auxiliary_kind_has_a_reporting_name(target: AuxiliaryTargetConfiguration) -> None:
    # reporting pulls in the quantization runtime, which is node-only.
    reporting = pytest.importorskip('src.training.reporting')
    (head,) = build_training_target_layout(ACTION_SIZE, (target,)).auxiliary_heads

    name = reporting._auxiliary_name(0, head)

    assert isinstance(name, str) and name.startswith('0-')


@pytest.mark.parametrize('target', EVERY_TARGET, ids=lambda target: target.kind)
def test_every_auxiliary_kind_captures_a_training_distribution(target: AuxiliaryTargetConfiguration) -> None:
    (head,) = build_training_target_layout(ACTION_SIZE, (target,)).auxiliary_heads
    rows = 4
    width = ACTION_SIZE if target.kind in ('next_policy', 'legal_moves') else 1
    aux_target = torch.full((rows, width), 1.0 / ACTION_SIZE if target.kind == 'next_policy' else 0.25)
    legal_action_ids = torch.arange(ACTION_SIZE).repeat(rows, 1)
    batch = TrainingBatch(
        states=torch.zeros(rows, 1),
        policy_targets=torch.full((rows, ACTION_SIZE), 1.0 / ACTION_SIZE),
        policy_legal_action_ids=legal_action_ids,
        wdl_targets=torch.full((rows, 3), 1.0 / 3.0),
        root_values=torch.zeros(rows),
        auxiliary_targets=(aux_target,),
        auxiliary_legal_action_ids=(legal_action_ids,),
        auxiliary_eligibility=(torch.ones(rows, dtype=torch.bool),),
        sample_weights=torch.ones(rows),
        source_model_generations=torch.zeros(rows, dtype=torch.long),
        source_created_at_seconds=torch.zeros(rows),
    )
    output = TrainingModelOutput(
        policy_logits=torch.zeros(rows, ACTION_SIZE),
        wdl_logits=torch.zeros(rows, 3),
        auxiliary_logits=(torch.zeros(rows, width),),
        features=torch.zeros(rows, 1),
    )
    objective = ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=resolve_auxiliary_losses((target,), model_generation=0),
    )

    snapshot = capture_training_distributions(output, batch, objective, source_generation=1, captured_at_seconds=1.0)

    assert snapshot.auxiliary[0] is not None
    assert snapshot.auxiliary[0].kind == head.kind
