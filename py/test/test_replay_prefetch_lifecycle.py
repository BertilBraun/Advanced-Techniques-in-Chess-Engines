from __future__ import annotations

import sys
from collections.abc import Generator
from threading import Event

import pytest
import torch
from src.replay.batch_loader import MappedReplayBatchLoader, PrefetchedReplayBatches, _PrefetchedBatch
from src.training.batch import TrainingBatch
from tools.benchmark_supervised_testbed import parse_arguments


def _batch(value: float) -> TrainingBatch:
    return TrainingBatch(
        states=torch.full((1, 1), value),
        policy_targets=torch.ones((1, 1)),
        policy_legal_action_ids=torch.zeros((1, 1), dtype=torch.int64),
        wdl_targets=torch.tensor(((0.0, 1.0, 0.0),)),
        root_values=torch.zeros(1),
        auxiliary_targets=(),
        auxiliary_legal_action_ids=(),
        auxiliary_eligibility=(),
        sample_weights=torch.ones(1),
        source_model_generations=torch.zeros(1, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(1, dtype=torch.float64),
    )


class _FakeReplayBatchLoader(MappedReplayBatchLoader[object]):
    def __init__(self, batch_count: int, failure_at: int | None = None) -> None:
        self.pin_memory = False
        self.batch_count = batch_count
        self.failure_at = failure_at
        self.started_count = 0
        self.preparation_started = Event()
        self.all_batches_started = Event()
        self.batch_started = tuple(Event() for _ in range(batch_count))
        self.generator_closed = Event()

    def _prepared_batches(self) -> Generator[TrainingBatch, None, None]:
        try:
            for index in range(self.batch_count):
                if index == self.failure_at:
                    raise ValueError('forced producer failure')
                self.started_count += 1
                self.preparation_started.set()
                self.batch_started[index].set()
                if self.started_count == self.batch_count:
                    self.all_batches_started.set()
                yield _batch(float(index))
        finally:
            self.generator_closed.set()


@pytest.mark.parametrize('depth', (1, 2, 4, 8))
def test_cpu_prefetch_is_ordered_bounded_and_closes_after_natural_exhaustion(depth: int) -> None:
    loader = _FakeReplayBatchLoader(batch_count=depth + 2)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=depth)
    assert loader.batch_started[depth - 1].wait(timeout=5.0)
    assert not loader.batch_started[depth].is_set()

    values = [float(batch.states[0, 0].item()) for batch in batches]

    assert values == [float(index) for index in range(depth + 2)]
    assert batches.closed
    assert loader.generator_closed.is_set()
    batches.close()


def test_cpu_prefetch_preserves_producer_failure_and_closes() -> None:
    loader = _FakeReplayBatchLoader(batch_count=3, failure_at=1)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=2)
    assert float(next(batches).states[0, 0].item()) == 0.0

    with pytest.raises(RuntimeError, match='Replay batch prefetch failed') as raised:
        next(batches)

    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == 'forced producer failure'
    assert batches.closed
    assert loader.generator_closed.is_set()


def test_next_closes_everything_and_preserves_transfer_visibility_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _FakeReplayBatchLoader(batch_count=2)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=1)
    synchronization_error = ValueError('forced transfer synchronization failure')

    def fail_synchronization(prefetched: _PrefetchedBatch) -> None:
        del prefetched
        raise synchronization_error

    monkeypatch.setattr(PrefetchedReplayBatches, '_make_transfer_visible', staticmethod(fail_synchronization))

    with pytest.raises(RuntimeError, match='Replay batch prefetch failed') as raised:
        next(batches)

    assert raised.value.__cause__ is synchronization_error
    assert batches.closed
    assert loader.generator_closed.is_set()


def test_close_finishes_cleanup_after_slot_transfer_sync_failure() -> None:
    loader = _FakeReplayBatchLoader(batch_count=2)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=2)
    synchronization_error = ValueError('forced transfer synchronization failure')

    class _FailingPool:
        def close(self) -> None:
            raise synchronization_error

    batches._pinned_slots = _FailingPool()  # type: ignore[assignment]
    assert loader.all_batches_started.wait(timeout=5.0)

    with pytest.raises(RuntimeError, match='cleanup failed') as raised:
        batches.close()

    assert raised.value.__cause__ is synchronization_error
    assert batches.closed
    assert loader.generator_closed.is_set()


def test_context_exit_preserves_body_failure_when_slot_cleanup_also_fails() -> None:
    loader = _FakeReplayBatchLoader(batch_count=1)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=1)
    body_error = ValueError('body failure')

    class _FailingPool:
        def close(self) -> None:
            raise RuntimeError('forced transfer synchronization failure')

    batches._pinned_slots = _FailingPool()  # type: ignore[assignment]

    with pytest.raises(ValueError, match='body failure') as raised:
        with batches:
            raise body_error

    assert raised.value is body_error
    assert batches.closed
    assert loader.generator_closed.is_set()


def test_next_closes_everything_and_preserves_scheduling_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _FakeReplayBatchLoader(batch_count=2)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=1)
    scheduling_error = RuntimeError('forced scheduling failure')

    def fail_scheduling(function: object) -> None:
        del function
        raise scheduling_error

    monkeypatch.setattr(batches._executor, 'submit', fail_scheduling)

    with pytest.raises(RuntimeError, match='Replay batch prefetch failed') as raised:
        next(batches)

    assert raised.value.__cause__ is scheduling_error
    assert batches.closed
    assert loader.generator_closed.is_set()


@pytest.mark.parametrize('depth', (1, 2, 4, 8))
def test_supervised_benchmark_accepts_supported_prefetch_depths(
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'benchmark_supervised_testbed.py',
            '--configuration',
            'experiment.yaml',
            '--train-store',
            'train.bin',
            '--holdout-store',
            'holdout.bin',
            '--cells',
            'cnn@0.01',
            '--replay-prefetch-depth',
            str(depth),
            '--output-path',
            'report.json',
        ],
    )

    assert parse_arguments().replay_prefetch_depth == depth
