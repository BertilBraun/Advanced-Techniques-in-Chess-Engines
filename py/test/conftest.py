from __future__ import annotations

import random

import numpy as np
import pytest
import torch

TEST_RANDOM_SEED = 20260921


@pytest.fixture(autouse=True)
def seed_random_number_generators() -> None:
    """Give every test the same generator state.

    Tests that draw probe positions from `torch.randn` inherit whatever state earlier tests left
    behind, so a bootstrap calibration that degenerates on one particular draw fails only for some
    orderings. Seeding per test makes such a failure reproducible instead of intermittent.
    """
    random.seed(TEST_RANDOM_SEED)
    np.random.seed(TEST_RANDOM_SEED)
    torch.manual_seed(TEST_RANDOM_SEED)
