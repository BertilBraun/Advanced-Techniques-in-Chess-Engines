from __future__ import annotations

import argparse
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from src.util.frozen_model import FrozenModel


class FixedModelSearchBudget(FrozenModel):
    kind: Literal['fixed_searches'] = 'fixed_searches'
    searches_per_move: int = Field(gt=0)
    parallel_searches: int = Field(gt=0)
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    outstanding_batches_per_worker: int = Field(gt=0, le=2)


class TimedModelSearchBudget(FrozenModel):
    kind: Literal['move_time'] = 'move_time'
    seconds_per_move: int = Field(ge=1, le=30)
    parallel_searches: int = Field(gt=0)
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    outstanding_batches_per_worker: int = Field(gt=0, le=2)


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


def model_search_budget(namespace: argparse.Namespace) -> FixedModelSearchBudget | TimedModelSearchBudget:
    if namespace.model_searches is not None:
        return FixedModelSearchBudget(
            searches_per_move=namespace.model_searches,
            parallel_searches=1 if namespace.parallel_searches is None else namespace.parallel_searches,
            inference_workers=1 if namespace.inference_workers is None else namespace.inference_workers,
            inference_batch_size=namespace.inference_batch_size,
            outstanding_batches_per_worker=(
                1 if namespace.outstanding_batches is None else namespace.outstanding_batches
            ),
        )
    if namespace.parallel_searches is None:
        raise ValueError('Timed search budgets require an explicit --parallel-searches value.')
    return TimedModelSearchBudget(
        seconds_per_move=namespace.model_move_time_seconds,
        parallel_searches=namespace.parallel_searches,
        inference_workers=2 if namespace.inference_workers is None else namespace.inference_workers,
        inference_batch_size=namespace.inference_batch_size,
        outstanding_batches_per_worker=2 if namespace.outstanding_batches is None else namespace.outstanding_batches,
    )
