import torch
from torch import Tensor


def scalar_to_wdl(values: Tensor) -> Tensor:
    """Convert expected scores in [-1, 1] to soft win/draw/loss targets."""
    squeezed_values = values.squeeze(-1) if values.ndim > 1 and values.shape[-1] == 1 else values
    bounded_values = torch.clamp(squeezed_values, min=-1.0, max=1.0)
    wins = torch.clamp(bounded_values, min=0.0)
    losses = torch.clamp(-bounded_values, min=0.0)
    draws = 1.0 - torch.abs(bounded_values)
    return torch.stack((wins, draws, losses), dim=-1)


def wdl_to_scalar(probabilities: Tensor) -> Tensor:
    """Return the expected score P(win) - P(loss) from WDL probabilities."""
    return probabilities[..., 0] - probabilities[..., 2]


def wdl_cross_entropy(probabilities: Tensor, targets: Tensor) -> Tensor:
    """Calculate cross entropy for already-normalized WDL probabilities."""
    return -(targets * torch.log(probabilities.clamp_min(1e-7))).sum(dim=-1).mean()
