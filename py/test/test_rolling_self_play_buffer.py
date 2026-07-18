import numpy as np
import pytest

from src.train.RollingSelfPlayBuffer import maximum_action_probability


def test_maximum_action_probability_uses_sparse_visit_counts() -> None:
    visit_counts = np.array(((12, 3), (24, 5), (48, 2)), dtype=np.uint16)

    assert maximum_action_probability(visit_counts) == pytest.approx(0.5)


def test_maximum_action_probability_rejects_zero_total() -> None:
    visit_counts = np.array(((12, 0),), dtype=np.uint16)

    with pytest.raises(ValueError, match='positive total'):
        maximum_action_probability(visit_counts)
