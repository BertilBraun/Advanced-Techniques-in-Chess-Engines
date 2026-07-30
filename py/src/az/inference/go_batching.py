from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from types import TracebackType

import numpy as np
import torch

import az_go_native as native
from src.az.games.go.configuration import GoGameConfiguration
from src.az.games.go.model import ResidualGoModel


@dataclass(frozen=True)
class InferenceBatchTelemetry:
    batches: int
    requests: int
    maximum_batch_size: int
    total_wait_microseconds: int
    cache_hits: int


class _PendingInference:
    def __init__(self, request: native.GoInferenceRequest) -> None:
        self.request = request
        self.condition = threading.Condition()
        self.result: native.InferenceResult | None = None
        self.failure: Exception | None = None
        self.submitted_at = time.monotonic_ns()


def _encoding_cache_key(encoding: native.GoEncoding) -> bytes:
    return np.asarray(encoding.values, dtype=np.int8).tobytes()


class GoInferenceBatchBroker:
    """Blocks native search callbacks while one device thread batches model forwards."""

    def __init__(
        self,
        model: ResidualGoModel,
        configuration: GoGameConfiguration,
        device: torch.device,
        maximum_batch_size: int,
        maximum_wait_microseconds: int,
        maximum_pending_batches: int,
        cache_capacity: int,
    ) -> None:
        if maximum_batch_size <= 0 or maximum_pending_batches <= 0:
            raise ValueError('Inference batch size and pending-batch capacity must be positive.')
        if maximum_wait_microseconds < 0:
            raise ValueError('Maximum inference wait cannot be negative.')
        if cache_capacity < 0:
            raise ValueError('Inference cache capacity cannot be negative.')
        self._model = model.to(device).eval()
        self._configuration = configuration
        self._device = device
        self._maximum_batch_size = maximum_batch_size
        self._maximum_wait_seconds = maximum_wait_microseconds / 1_000_000
        self._capacity = threading.BoundedSemaphore(maximum_pending_batches * maximum_batch_size)
        self._cache_capacity = cache_capacity
        self._cache: OrderedDict[bytes, tuple[tuple[float, ...], float]] = OrderedDict()
        self._pending: deque[_PendingInference] = deque()
        self._condition = threading.Condition()
        self._stopping = False
        self._failure: Exception | None = None
        self._batches = 0
        self._requests = 0
        self._maximum_observed_batch = 0
        self._total_wait_microseconds = 0
        self._cache_hits = 0
        self._thread = threading.Thread(target=self._serve, name='go-inference-broker', daemon=False)
        self._thread.start()

    @property
    def telemetry(self) -> InferenceBatchTelemetry:
        with self._condition:
            return InferenceBatchTelemetry(
                batches=self._batches,
                requests=self._requests,
                maximum_batch_size=self._maximum_observed_batch,
                total_wait_microseconds=self._total_wait_microseconds,
                cache_hits=self._cache_hits,
            )

    @property
    def model(self) -> ResidualGoModel:
        return self._model

    def take_telemetry(self) -> InferenceBatchTelemetry:
        with self._condition:
            telemetry = InferenceBatchTelemetry(
                batches=self._batches,
                requests=self._requests,
                maximum_batch_size=self._maximum_observed_batch,
                total_wait_microseconds=self._total_wait_microseconds,
                cache_hits=self._cache_hits,
            )
            self._batches = 0
            self._requests = 0
            self._maximum_observed_batch = 0
            self._total_wait_microseconds = 0
            self._cache_hits = 0
            return telemetry

    def evaluate(self, request: native.GoInferenceRequest) -> native.InferenceResult:
        cache_key = _encoding_cache_key(request.encoding)
        with self._condition:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                self._cache_hits += 1
                return native.InferenceResult(
                    request.request_id,
                    list(cached[0]),
                    cached[1],
                )
        self._capacity.acquire()
        pending = _PendingInference(request)
        with self._condition:
            if self._stopping:
                self._capacity.release()
                raise RuntimeError('Inference broker is stopping.')
            if self._failure is not None:
                self._capacity.release()
                raise RuntimeError('Inference broker failed.') from self._failure
            self._pending.append(pending)
            self._condition.notify()
        with pending.condition:
            while pending.result is None and pending.failure is None:
                pending.condition.wait()
        self._capacity.release()
        if pending.failure is not None:
            raise RuntimeError('Batched Go inference failed.') from pending.failure
        if pending.result is None:
            raise AssertionError('Inference request completed without a result.')
        return pending.result

    def close(self) -> InferenceBatchTelemetry:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join()
        if self._failure is not None:
            raise RuntimeError('Inference broker failed.') from self._failure
        return self.telemetry

    def __enter__(self) -> GoInferenceBatchBroker:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def _serve(self) -> None:
        batch: tuple[_PendingInference, ...] = ()
        try:
            while True:
                batch = self._next_batch()
                if not batch:
                    return
                self._run_batch(batch)
        except Exception as error:
            with self._condition:
                self._failure = error
                pending = tuple(self._pending)
                self._pending.clear()
            self._fail((*batch, *pending), error)

    def _next_batch(self) -> tuple[_PendingInference, ...]:
        with self._condition:
            while not self._pending and not self._stopping:
                self._condition.wait()
            if not self._pending:
                return ()
            deadline = time.monotonic() + self._maximum_wait_seconds
            while len(self._pending) < self._maximum_batch_size and not self._stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            count = min(len(self._pending), self._maximum_batch_size)
            return tuple(self._pending.popleft() for _ in range(count))

    def _run_batch(self, batch: tuple[_PendingInference, ...]) -> None:
        shape = (
            self._configuration.input_plane_count,
            self._configuration.board_size,
            self._configuration.board_size,
        )
        arrays: list[np.ndarray] = []
        for pending in batch:
            encoding = pending.request.encoding
            if (
                encoding.planes != shape[0]
                or encoding.board_size != shape[1]
                or len(encoding.values) != int(np.prod(shape))
            ):
                raise ValueError('Native inference encoding does not match the Go model.')
            arrays.append(np.asarray(encoding.values, dtype=np.float32).reshape(shape))
        inputs = torch.from_numpy(np.stack(arrays)).to(self._device)
        with torch.inference_mode():
            outputs = self._model(inputs)
            policies = torch.softmax(outputs.policy_logits, dim=1).cpu().tolist()
            values = outputs.value.cpu().tolist()
        completed_at = time.monotonic_ns()
        with self._condition:
            self._batches += 1
            self._requests += len(batch)
            self._maximum_observed_batch = max(self._maximum_observed_batch, len(batch))
            self._total_wait_microseconds += sum((completed_at - pending.submitted_at) // 1_000 for pending in batch)
        for pending, policy, value in zip(batch, policies, values, strict=True):
            if self._cache_capacity:
                cache_key = _encoding_cache_key(pending.request.encoding)
                with self._condition:
                    self._cache[cache_key] = (tuple(policy), value)
                    self._cache.move_to_end(cache_key)
                    while len(self._cache) > self._cache_capacity:
                        self._cache.popitem(last=False)
            with pending.condition:
                pending.result = native.InferenceResult(pending.request.request_id, policy, value)
                pending.condition.notify()

    @staticmethod
    def _fail(pending_requests: tuple[_PendingInference, ...], error: Exception) -> None:
        for pending in pending_requests:
            with pending.condition:
                pending.failure = error
                pending.condition.notify()
