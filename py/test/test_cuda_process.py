import os

import pytest

from src.cluster.CudaProcess import start_process_on_cuda_device


class _EnvironmentCapturingProcess:
    def __init__(self) -> None:
        self.visible_cuda_device: str | None = None

    def start(self) -> None:
        self.visible_cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES')


@pytest.mark.parametrize('previous_value', [None, '0,1'])
def test_cuda_process_inherits_only_assigned_device_and_restores_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
    previous_value: str | None,
) -> None:
    if previous_value is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', previous_value)
    process = _EnvironmentCapturingProcess()

    start_process_on_cuda_device(process, physical_device_id=3)

    assert process.visible_cuda_device == '3'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == previous_value


def test_cuda_process_restores_parent_environment_when_start_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingProcess:
        def start(self) -> None:
            raise RuntimeError('spawn failed')

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')

    with pytest.raises(RuntimeError, match='spawn failed'):
        start_process_on_cuda_device(FailingProcess(), physical_device_id=2)

    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
