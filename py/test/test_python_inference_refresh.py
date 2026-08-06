from __future__ import annotations

from os import PathLike

import pytest
import torch

from src.cluster.InferenceClient import InferenceClient
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


def _client(model: _FakeModel) -> InferenceClient:
    client = object.__new__(InferenceClient)
    client.network_args = TRAINING_ARGS.network
    client.save_path = '.'
    client.model = model
    client.model_version = 3
    client.device = torch.device('cpu')
    client.dtype = torch.float32
    return client


def test_python_inference_refresh_prepares_before_transactional_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model = _FakeModel('previous')
    updated_model = _FakeModel('updated')
    client = _client(previous_model)

    def load_updated_model(
        _model_path: str | PathLike[str],
        _network_args: NetworkParams,
        _device: torch.device,
    ) -> _FakeModel:
        assert client.model is previous_model
        return updated_model

    monkeypatch.setattr('src.cluster.InferenceClient.load_model', load_updated_model)

    client.refresh_model(4, 'updated.pt')

    assert client.model is updated_model
    assert client.model_version == 4
    assert updated_model.preparation_steps == ['to', 'disable_auto_grad', 'eval', 'fuse_model']


def test_python_inference_refresh_failure_preserves_previous_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model = _FakeModel('previous')
    client = _client(previous_model)
    attempts = 0

    def fail_load(
        _model_path: str | PathLike[str],
        _network_args: NetworkParams,
        _device: torch.device,
    ) -> _FakeModel:
        nonlocal attempts
        attempts += 1
        raise RuntimeError('broken checkpoint')

    monkeypatch.setattr('src.cluster.InferenceClient.load_model', fail_load)
    monkeypatch.setattr('src.cluster.InferenceClient.sleep', lambda _seconds: None)

    with pytest.raises(RuntimeError, match='Failed to load model after 5 retries'):
        client.refresh_model(4, 'broken.pt')

    assert attempts == 5
    assert client.model is previous_model
    assert client.model_version == 3
