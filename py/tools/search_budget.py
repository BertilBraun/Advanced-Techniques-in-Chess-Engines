from __future__ import annotations

import argparse
from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from src.evaluation.configuration import EvaluationTreeSearchOverrides
from src.self_play.configuration import (
    ParentValueFirstPlayUrgencyConfiguration,
    ReducedParentValueFirstPlayUrgencyConfiguration,
    ZeroFirstPlayUrgencyConfiguration,
)
from src.util.frozen_model import FrozenModel
from src.util.generation_schedule import ConstantSchedule

_DEFAULT_EXPLORATION_CONSTANT = 1.0


class FixedModelSearchBudget(FrozenModel):
    kind: Literal['fixed_searches'] = 'fixed_searches'
    searches_per_move: int = Field(gt=0)
    parallel_searches: int = Field(gt=0)
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    outstanding_batches_per_worker: int = Field(gt=0, le=2)
    exploration_constant: float = Field(default=_DEFAULT_EXPLORATION_CONSTANT, gt=0.0)
    tree_search: EvaluationTreeSearchOverrides | None = None


class TimedModelSearchBudget(FrozenModel):
    kind: Literal['move_time'] = 'move_time'
    seconds_per_move: int = Field(ge=1, le=30)
    parallel_searches: int = Field(gt=0)
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    outstanding_batches_per_worker: int = Field(gt=0, le=2)
    exploration_constant: float = Field(default=_DEFAULT_EXPLORATION_CONSTANT, gt=0.0)
    tree_search: EvaluationTreeSearchOverrides | None = None


ModelSearchBudget: TypeAlias = Annotated[
    FixedModelSearchBudget | TimedModelSearchBudget,
    Field(discriminator='kind'),
]


def add_model_search_budget_arguments(parser: argparse.ArgumentParser) -> None:
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--model-searches', type=int)
    budget.add_argument('--model-move-time-seconds', type=int)
    parser.add_argument('--parallel-searches', type=int)
    parser.add_argument('--inference-workers', type=int)
    parser.add_argument('--inference-batch-size', default=64, type=int)
    parser.add_argument('--outstanding-batches', type=int)
    parser.add_argument('--exploration-constant', default=_DEFAULT_EXPLORATION_CONSTANT, type=float)
    parser.add_argument('--first-play-urgency', choices=('zero', 'parent_value', 'reduced_parent_value'))
    parser.add_argument('--first-play-urgency-reduction', type=float)
    parser.add_argument('--virtual-loss-weight', type=float)
    parser.add_argument('--search-value-discount-per-ply', type=float)


def _tree_search_overrides(namespace: argparse.Namespace) -> EvaluationTreeSearchOverrides | None:
    requested = (
        namespace.first_play_urgency,
        namespace.virtual_loss_weight,
        namespace.search_value_discount_per_ply,
    )
    if all(value is None for value in requested):
        if namespace.first_play_urgency_reduction is not None:
            raise ValueError('--first-play-urgency-reduction requires --first-play-urgency.')
        return None
    if any(value is None for value in requested):
        raise ValueError(
            'Tree-search overrides are all-or-nothing: pass --first-play-urgency, --virtual-loss-weight '
            'and --search-value-discount-per-ply together so every arm records what it ran.'
        )
    match namespace.first_play_urgency:
        case 'zero':
            first_play_urgency = ZeroFirstPlayUrgencyConfiguration()
        case 'parent_value':
            first_play_urgency = ParentValueFirstPlayUrgencyConfiguration()
        case 'reduced_parent_value':
            if namespace.first_play_urgency_reduction is None:
                raise ValueError('reduced_parent_value FPU requires --first-play-urgency-reduction.')
            first_play_urgency = ReducedParentValueFirstPlayUrgencyConfiguration(
                reduction=ConstantSchedule[float](value=namespace.first_play_urgency_reduction),
            )
        case _:
            raise ValueError(f'Unknown first-play-urgency kind: {namespace.first_play_urgency!r}')
    if namespace.first_play_urgency != 'reduced_parent_value' and namespace.first_play_urgency_reduction is not None:
        raise ValueError('Only reduced_parent_value FPU accepts a reduction.')
    return EvaluationTreeSearchOverrides(
        first_play_urgency=first_play_urgency,
        virtual_loss_weight=namespace.virtual_loss_weight,
        value_discount_per_ply=namespace.search_value_discount_per_ply,
    )


def model_search_budget(namespace: argparse.Namespace) -> FixedModelSearchBudget | TimedModelSearchBudget:
    tree_search = _tree_search_overrides(namespace)
    if namespace.model_searches is not None:
        return FixedModelSearchBudget(
            searches_per_move=namespace.model_searches,
            parallel_searches=1 if namespace.parallel_searches is None else namespace.parallel_searches,
            inference_workers=1 if namespace.inference_workers is None else namespace.inference_workers,
            inference_batch_size=namespace.inference_batch_size,
            outstanding_batches_per_worker=(
                1 if namespace.outstanding_batches is None else namespace.outstanding_batches
            ),
            exploration_constant=namespace.exploration_constant,
            tree_search=tree_search,
        )
    if namespace.parallel_searches is None:
        raise ValueError('Timed search budgets require an explicit --parallel-searches value.')
    if tree_search is not None:
        raise ValueError('Timed budgets run the analysis engine, which cannot express tree-search overrides.')
    return TimedModelSearchBudget(
        seconds_per_move=namespace.model_move_time_seconds,
        parallel_searches=namespace.parallel_searches,
        inference_workers=2 if namespace.inference_workers is None else namespace.inference_workers,
        inference_batch_size=namespace.inference_batch_size,
        outstanding_batches_per_worker=2 if namespace.outstanding_batches is None else namespace.outstanding_batches,
        exploration_constant=namespace.exploration_constant,
    )
