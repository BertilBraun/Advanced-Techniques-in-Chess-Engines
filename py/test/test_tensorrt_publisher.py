from __future__ import annotations

import numpy as np
import pytest
from tools.publish_tensorrt_engine import _policy_distribution_agreement


def test_policy_distribution_agreement_is_exact_for_equal_logits() -> None:
    logits = np.array(((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)), dtype=np.float32)

    top1_agreement, mean_divergence, maximum_divergence = _policy_distribution_agreement(logits, logits)

    assert top1_agreement == 1.0
    assert mean_divergence == pytest.approx(0.0, abs=1e-7)
    assert maximum_divergence == pytest.approx(0.0, abs=1e-7)


def test_policy_distribution_agreement_detects_changed_policy() -> None:
    reference = np.array(((5.0, 0.0, -1.0), (0.0, 5.0, -1.0)), dtype=np.float32)
    candidate = np.array(((0.0, 5.0, -1.0), (0.0, 5.0, -1.0)), dtype=np.float32)

    top1_agreement, mean_divergence, maximum_divergence = _policy_distribution_agreement(reference, candidate)

    assert top1_agreement == 0.5
    assert mean_divergence > 0.0
    assert maximum_divergence > mean_divergence
