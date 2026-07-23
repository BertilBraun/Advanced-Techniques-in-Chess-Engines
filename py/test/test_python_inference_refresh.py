from __future__ import annotations

from os import PathLike

import pytest
import numpy as np
import torch

from src.cluster.InferenceClient import InferenceClient
from src.cluster.NonCachingInferenceClient import NonCachingInferenceClient
from src.settings import TRAINING_ARGS
from src.train.TrainingArgs import NetworkParams


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.preparation_steps: list[str] = []

    def to(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        non_blocking: bool,
    ) -> _FakeModel:
        assert dtype is torch.float32
        assert device.type == 'cpu'
        assert non_blocking
        self.preparation_steps.append('to')
        return self

    def disable_auto_grad(self) -> None:
        self.preparation_steps.append('disable_auto_grad')

    def eval(self) -> _FakeModel:
        self.preparation_steps.append('eval')
        return self

    def fuse_model(self) -> None:
        self.preparation_steps.append('fuse_model')


InferenceClientType = type[InferenceClient] | type[NonCachingInferenceClient]
InferenceClientInstance = InferenceClient | NonCachingInferenceClient


def _client(client_type: InferenceClientType, model: _FakeModel) -> InferenceClientInstance:
    client = object.__new__(client_type)
    client.network_args = TRAINING_ARGS.network
    client.save_path = '.'
    client.model = model
    client.model_version = 3
    client.device = torch.device('cpu')
    client.dtype = torch.float32
    if isinstance(client, InferenceClient):
        client.inference_cache = {7: (np.array([[1.0, 0.5]]), 0.5)}
        client.total_hits = 0
        client.total_evals = 0
    return client


@pytest.mark.parametrize('client_type', (InferenceClient, NonCachingInferenceClient))
def test_python_inference_refresh_prepares_before_transactional_swap(
    monkeypatch: pytest.MonkeyPatch,
    client_type: InferenceClientType,
) -> None:
    previous_model = _FakeModel('previous')
    updated_model = _FakeModel('updated')
    client = _client(client_type, previous_model)

    def load_updated_model(
        _model_path: str | PathLike[str],
        _network_args: NetworkParams,
        _device: torch.device,
    ) -> _FakeModel:
        assert client.model is previous_model
        return updated_model

    module_name = client_type.__module__
    monkeypatch.setattr(f'{module_name}.load_model', load_updated_model)

    client.refresh_model(4, 'updated.pt')

    assert client.model is updated_model
    assert client.model_version == 4
    assert updated_model.preparation_steps == ['to', 'disable_auto_grad', 'eval', 'fuse_model']
    if isinstance(client, InferenceClient):
        assert client.inference_cache == {}


@pytest.mark.parametrize('client_type', (InferenceClient, NonCachingInferenceClient))
def test_python_inference_refresh_failure_preserves_previous_model(
    monkeypatch: pytest.MonkeyPatch,
    client_type: InferenceClientType,
) -> None:
    previous_model = _FakeModel('previous')
    client = _client(client_type, previous_model)
    previous_cache = client.inference_cache if isinstance(client, InferenceClient) else None
    attempts = 0

    def fail_load(
        _model_path: str | PathLike[str],
        _network_args: NetworkParams,
        _device: torch.device,
    ) -> _FakeModel:
        nonlocal attempts
        attempts += 1
        raise RuntimeError('broken checkpoint')

    module_name = client_type.__module__
    monkeypatch.setattr(f'{module_name}.load_model', fail_load)
    monkeypatch.setattr(f'{module_name}.sleep', lambda _seconds: None)

    with pytest.raises(RuntimeError, match='Failed to load model after 5 retries'):
        client.refresh_model(4, 'broken.pt')

    assert attempts == 5
    assert client.model is previous_model
    assert client.model_version == 3
    if isinstance(client, InferenceClient):
        assert client.inference_cache is previous_cache
