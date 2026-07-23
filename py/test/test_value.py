import torch

from src.value import wdl_to_scalar


def test_wdl_expected_score_uses_win_minus_loss() -> None:
    probabilities = torch.tensor(((0.7, 0.2, 0.1), (0.1, 0.4, 0.5)))

    torch.testing.assert_close(wdl_to_scalar(probabilities), torch.tensor((0.6, -0.4)))
