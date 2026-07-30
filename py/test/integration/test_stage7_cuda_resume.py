from __future__ import annotations

import pytest
import torch

from src.az.training.optimizer import (
    restore_assigned_cuda_random_state,
    restore_gradient_scaler,
    serialize_assigned_cuda_random_state,
    serialize_gradient_scaler,
)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware is not configured')
def test_cuda_rng_streams_and_gradient_scaler_restore() -> None:
    torch.cuda.manual_seed_all(701)
    scaler = torch.amp.GradScaler(device='cuda', enabled=True)
    device = torch.device('cuda:0')
    random_states = serialize_assigned_cuda_random_state(device)
    scaler_state = serialize_gradient_scaler(scaler)
    expected = torch.rand(8, device='cuda')
    torch.cuda.manual_seed_all(999)
    restore_assigned_cuda_random_state(random_states, device)
    restored_scaler = torch.amp.GradScaler(device='cuda', enabled=True)
    restore_gradient_scaler(restored_scaler, scaler_state)
    actual = torch.rand(8, device='cuda')

    assert torch.equal(actual, expected)
    assert restored_scaler.state_dict() == scaler.state_dict()
