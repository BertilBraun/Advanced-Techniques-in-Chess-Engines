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
    def __init__(self, batch_count: int) -> None:
        self.batch_count = batch_count
        self.started_count = 0
        self.preparation_started = Event()
        self.all_batches_started = Event()
        self.generator_closed = Event()

    def _prepared_batches(self) -> Generator[TrainingBatch, None, None]:
        try:
            for index in range(self.batch_count):
                self.started_count += 1
                self.preparation_started.set()
                if self.started_count == self.batch_count:
                    self.all_batches_started.set()
                yield _batch(float(index))
        finally:
            self.generator_closed.set()


def test_next_closes_everything_and_preserves_transfer_sync_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _FakeReplayBatchLoader(batch_count=2)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=1)
    synchronization_error = ValueError('forced transfer synchronization failure')

    def fail_synchronization(prefetched: _PrefetchedBatch) -> None:
        del prefetched
        raise synchronization_error

    monkeypatch.setattr(PrefetchedReplayBatches, '_synchronize_transfer', staticmethod(fail_synchronization))

    with pytest.raises(RuntimeError, match='Replay batch prefetch failed') as raised:
        next(batches)

    assert raised.value.__cause__ is synchronization_error
    assert batches.closed
    assert loader.generator_closed.is_set()


def test_close_finishes_cleanup_after_transfer_sync_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _FakeReplayBatchLoader(batch_count=2)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=2)
    synchronization_error = ValueError('forced transfer synchronization failure')
    synchronized_count = 0

    def fail_synchronization(prefetched: _PrefetchedBatch) -> None:
        nonlocal synchronized_count
        del prefetched
        synchronized_count += 1
        raise synchronization_error

    monkeypatch.setattr(PrefetchedReplayBatches, '_synchronize_transfer', staticmethod(fail_synchronization))
    assert loader.all_batches_started.wait(timeout=5.0)

    with pytest.raises(RuntimeError, match='cleanup failed') as raised:
        batches.close()

    assert raised.value.__cause__ is synchronization_error
    assert synchronized_count == 2
    assert batches.closed
    assert loader.generator_closed.is_set()


def test_context_exit_preserves_body_failure_when_transfer_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _FakeReplayBatchLoader(batch_count=1)
    batches = PrefetchedReplayBatches(loader, torch.device('cpu'), uses_cuda=False, depth=1)
    body_error = ValueError('body failure')

    def fail_synchronization(prefetched: _PrefetchedBatch) -> None:
        del prefetched
        raise RuntimeError('forced transfer synchronization failure')

    monkeypatch.setattr(PrefetchedReplayBatches, '_synchronize_transfer', staticmethod(fail_synchronization))

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
