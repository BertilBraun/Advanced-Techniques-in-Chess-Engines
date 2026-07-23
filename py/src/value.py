from torch import Tensor

from src.self_play.value_target import FinalOutcome


WIN_INDEX = int(FinalOutcome.WIN)
LOSS_INDEX = int(FinalOutcome.LOSS)


def wdl_to_scalar(probabilities: Tensor) -> Tensor:
    """Return the expected score P(win) - P(loss) from WDL probabilities."""
    return probabilities[..., WIN_INDEX] - probabilities[..., LOSS_INDEX]
