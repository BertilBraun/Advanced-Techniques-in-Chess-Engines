from __future__ import annotations

import torch
from torch import nn

from src.az.config.training import (
    AdamWOptimizerConfiguration,
    LearningRateStage,
    PiecewiseLearningRate,
)
from src.az.training.optimizer import LearningRateController, create_optimizer


def test_piecewise_learning_rate_advances_at_exact_optimizer_steps() -> None:
    model = nn.Linear(2, 1)
    optimizer_configuration = AdamWOptimizerConfiguration(
        kind='adamw',
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=0,
    )
    schedule = PiecewiseLearningRate(
        kind='piecewise',
        stages=(
            LearningRateStage(start_optimizer_step=0, multiplier=1),
            LearningRateStage(start_optimizer_step=2, multiplier=0.1),
        ),
    )
    optimizer = create_optimizer(model, optimizer_configuration)
    controller = LearningRateController(optimizer, 0.01, schedule)

    assert controller.state.current_learning_rate == 0.01
    assert controller.advance().current_learning_rate == 0.01
    resumed = LearningRateController(optimizer, 0.01, schedule, controller.state)
    assert resumed.advance().current_learning_rate == 0.001
    assert torch.isclose(torch.tensor(optimizer.param_groups[0]['lr']), torch.tensor(0.001))
