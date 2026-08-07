import torch
from torch import Tensor

from src.self_play.value_target import FinalOutcome


WIN_INDEX = int(FinalOutcome.WIN)
LOSS_INDEX = int(FinalOutcome.LOSS)


def wdl_to_scalar(probabilities: Tensor) -> Tensor:
    """Return the expected score P(win) - P(loss) from WDL probabilities."""
    return probabilities[..., WIN_INDEX] - probabilities[..., LOSS_INDEX]


def scalar_to_wdl(scores: Tensor) -> Tensor:
    """Represent scores in [-1, 1] with residual probability shared across WDL."""
    wins = torch.clamp(scores, min=0.0)
    losses = torch.clamp(-scores, min=0.0)
    remainders = 1.0 - torch.abs(scores)
    return torch.stack((wins + remainders / 3, remainders / 3, losses + remainders / 3), dim=-1)
