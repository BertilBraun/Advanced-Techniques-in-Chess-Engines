from pathlib import Path

import pytest
import torch
import torch.distributed as distributed


pytest.importorskip('GPUtil')

from src.cluster.TrainerProcess import (
    DistributedTrainingError,
    RankFailure,
    RankReady,
    TrainerProcess,
    _wrap_distributed_model,
    available_tcp_port,
    is_rank_zero,
)
from src.settings import TRAINING_ARGS
from src.util.save_paths import create_model, create_optimizer, load_model_and_optimizer, save_model_and_optimizer


class _Connection:
    def __init__(self, response: RankFailure | None) -> None:
        self.response = response
        self.closed = False

    def poll(self) -> bool:
        return self.response is not None

    def recv(self) -> RankFailure:
        assert self.response is not None
        response = self.response
        self.response = None
        return response

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self) -> None:
        self.terminated = False

    def is_alive(self) -> bool:
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        assert timeout is None or timeout >= 0

    def kill(self) -> None:
        self.terminated = True


def test_only_rank_zero_owns_trainer_side_effects() -> None:
    assert is_rank_zero(0)
    assert not is_rank_zero(1)
    assert not is_rank_zero(3)


def test_rank_failure_aborts_every_peer_and_preserves_remote_error() -> None:
    failure = RankFailure(
        rank=1,
        phase_id=None,
        exception_type='SyntheticFailure',
        message='original rank failure',
        formatted_traceback='remote traceback',
    )
    coordinator = TrainerProcess.__new__(TrainerProcess)
    coordinator.world_size = 2
    coordinator._connections = [_Connection(None), _Connection(failure)]
    coordinator._processes = [_Process(), _Process()]
    coordinator._failed = False

    with pytest.raises(DistributedTrainingError, match='original rank failure') as raised:
        coordinator._collect_responses(RankReady, phase_id=None)

    assert raised.value.failure is failure
    assert coordinator._failed
    assert coordinator._connections == []
    assert coordinator._processes == []


def test_unwrapped_checkpoint_remains_single_rank_compatible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initialization_method = f'tcp://127.0.0.1:{available_tcp_port()}'
    distributed.init_process_group(
        backend='gloo',
        init_method=initialization_method,
        rank=0,
        world_size=1,
    )
    try:
        device = torch.device('cpu')
        model = create_model(TRAINING_ARGS.network, device)
        optimizer = create_optimizer(model, TRAINING_ARGS.training.optimizer)
        wrapped_model = _wrap_distributed_model(model, device)

        assert any(key.startswith('module.') for key in wrapped_model.state_dict())
        assert all(not key.startswith('module.') for key in model.state_dict())

        save_model_and_optimizer(model, optimizer, 0, tmp_path)
        original_torch_load = torch.load
        optimizer_map_locations: list[torch.device] = []

        def load_checkpoint(
            path: str | Path,
            *,
            weights_only: bool,
            map_location: torch.device | None = None,
        ) -> object:
            if Path(path).name.startswith('optimizer_'):
                assert map_location is not None
                optimizer_map_locations.append(map_location)
            return original_torch_load(path, weights_only=weights_only, map_location=map_location)

        monkeypatch.setattr(torch, 'load', load_checkpoint)
        loaded_model, loaded_optimizer = load_model_and_optimizer(
            0,
            TRAINING_ARGS.network,
            device,
            tmp_path,
            TRAINING_ARGS.training.optimizer,
        )
    finally:
        distributed.destroy_process_group()

    assert loaded_model.state_dict().keys() == model.state_dict().keys()
    assert loaded_optimizer.state_dict()['param_groups'] == optimizer.state_dict()['param_groups']
    assert optimizer_map_locations == [device]
