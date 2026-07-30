from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import Tensor

native = pytest.importorskip('az_go_native', reason='focused native Go extension has not been built')

from src.az.games.go.model import GoModelOutput, ResidualGoModel
from src.az.inference.go_batching import GoInferenceBatchBroker
from test.unit.go_stage5_helpers import game_configuration, model_configuration


def test_concurrent_native_searches_share_actual_model_batch() -> None:
    game = game_configuration()
    model = ResidualGoModel(game, model_configuration())
    barrier = threading.Barrier(2)
    configuration = native.FixedPuctConfiguration(
        simulation_cap=2,
        exploration_constant=1.5,
        backup_discount=1.0,
        no_visited_child_value=0.0,
        action_temperature=0.0,
        root_noise_seed=1,
        action_sampling_seed=2,
        root_noise=native.RootNoiseConfiguration(False, 0.3, 0.0),
        tree_reuse=False,
    )
    with GoInferenceBatchBroker(
        model=model,
        configuration=game,
        device=torch.device('cpu'),
        maximum_batch_size=2,
        maximum_wait_microseconds=100_000,
        maximum_pending_batches=2,
        cache_capacity=0,
    ) as broker:

        def search(seed_offset: int) -> native.SearchResult:
            state = native.GoState(
                native.GoRules(game.board_size, game.komi_half_points, game.safety_ply_cap, game.history_length)
            )

            def evaluate(request: native.GoInferenceRequest) -> native.InferenceResult:
                if request.request_id == 0:
                    barrier.wait()
                return broker.evaluate(request)

            del seed_offset
            return native.search_go_fixed(state, evaluate, configuration)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(search, (0, 1)))
        telemetry = broker.telemetry

    assert all(result.telemetry.actual_simulations == 2 for result in results)
    assert telemetry.requests >= 4
    assert telemetry.maximum_batch_size == 2
    assert telemetry.total_wait_microseconds > 0


class _FailingGoModel(ResidualGoModel):
    def forward(self, inputs: Tensor) -> GoModelOutput:
        del inputs
        raise RuntimeError('injected inference failure')


class _CountingGoModel(ResidualGoModel):
    def __init__(self) -> None:
        game = game_configuration()
        super().__init__(game, model_configuration())
        self.forward_calls = 0

    def forward(self, inputs: Tensor) -> GoModelOutput:
        self.forward_calls += 1
        return super().forward(inputs)


def test_repeated_identical_encoding_uses_compact_inference_cache_key() -> None:
    game = game_configuration()
    model = _CountingGoModel()
    state = native.GoState(
        native.GoRules(game.board_size, game.komi_half_points, game.safety_ply_cap, game.history_length)
    )
    configuration = native.FixedPuctConfiguration(
        1,
        1.5,
        1.0,
        0.0,
        0.0,
        1,
        2,
        native.RootNoiseConfiguration(False, 0.3, 0.0),
        False,
    )
    with GoInferenceBatchBroker(
        model=model,
        configuration=game,
        device=torch.device('cpu'),
        maximum_batch_size=1,
        maximum_wait_microseconds=0,
        maximum_pending_batches=1,
        cache_capacity=8,
    ) as broker:

        def evaluate_twice(request: native.GoInferenceRequest) -> native.InferenceResult:
            broker.evaluate(request)
            return broker.evaluate(request)

        native.search_go_fixed(state, evaluate_twice, configuration)
        telemetry = broker.telemetry

    assert telemetry.requests == model.forward_calls
    assert telemetry.cache_hits >= 1


def test_broker_failure_wakes_all_blocked_native_searches() -> None:
    game = game_configuration()
    broker = GoInferenceBatchBroker(
        model=_FailingGoModel(game, model_configuration()),
        configuration=game,
        device=torch.device('cpu'),
        maximum_batch_size=2,
        maximum_wait_microseconds=100_000,
        maximum_pending_batches=2,
        cache_capacity=0,
    )
    barrier = threading.Barrier(2)
    configuration = native.FixedPuctConfiguration(
        1,
        1.5,
        1.0,
        0.0,
        0.0,
        1,
        2,
        native.RootNoiseConfiguration(False, 0.3, 0.0),
        False,
    )

    def failing_search(index: int) -> None:
        state = native.GoState(
            native.GoRules(game.board_size, game.komi_half_points, game.safety_ply_cap, game.history_length)
        )

        def evaluate(request: native.GoInferenceRequest) -> native.InferenceResult:
            barrier.wait()
            return broker.evaluate(request)

        del index
        native.search_go_fixed(state, evaluate, configuration)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(failing_search, index) for index in range(2))
        for future in futures:
            with pytest.raises(RuntimeError, match='Batched Go inference failed'):
                future.result(timeout=10)
    with pytest.raises(RuntimeError, match='Inference broker failed'):
        broker.close()
