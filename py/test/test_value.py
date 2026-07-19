import torch

from src.value import scalar_to_wdl, wdl_to_scalar


def test_scalar_wdl_conversion_preserves_expected_score() -> None:
    values = torch.tensor((-1.0, -0.25, 0.0, 0.4, 1.0))

    probabilities = scalar_to_wdl(values)

    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(len(values)))
    torch.testing.assert_close(wdl_to_scalar(probabilities), values)
