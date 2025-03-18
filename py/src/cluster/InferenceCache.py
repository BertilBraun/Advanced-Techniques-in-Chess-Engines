from __future__ import annotations
import time
from typing import TypeVar


import numpy as np
from sys import getsizeof

from src.Encoding import (
    MoveScore,
)
from src.settings import log_histogram, log_scalar
from src.util.log import LogLevel, log


T = TypeVar('T')
MoveList = np.ndarray  #  list[MoveScore]


class InferenceCache:
    def __init__(self) -> None:
        self.inference_cache: dict[int, tuple[MoveList, float]] = {}
        self.total_hits = 0
        self.total_evals = 0

        self.start_time = time.time()

    def filter(self, hashes: list[int], data: list[T]) -> tuple[list[int], list[T]]:
        not_cached: list[int] = []
        not_cached_data: list[T] = []

        enqueued: set[int] = set()

        for hash, value in zip(hashes, data):
            if hash not in self.inference_cache and hash not in enqueued:
                enqueued.add(hash)
                not_cached.append(hash)
                not_cached_data.append(value)
            else:
                self.total_hits += 1

        self.total_evals += len(hashes)

        return not_cached, not_cached_data

    def log_stats(self, iteration: int) -> None:
        if self.total_evals != 0:
            cache_hit_rate = (self.total_hits / self.total_evals) * 100
            log_scalar('inferences_per_second', self.total_evals / (time.time() - self.start_time), iteration)
            log_scalar('cache/hit_rate', cache_hit_rate, iteration)
            log_scalar('cache/unique_positions', len(self.inference_cache), iteration)
            log_histogram(
                'nn_output_value_distribution',
                np.array([v for _, v in self.inference_cache.values()]),
                iteration,
            )

            size_in_mb = 0
            for key, (policy, value) in self.inference_cache.items():
                size_in_mb += getsizeof(key) + getsizeof(value) + getsizeof(policy) + getsizeof(policy[0]) * len(policy)

            size_in_mb /= 1024 * 1024
            log_scalar('cache/size_mb', size_in_mb, iteration)
            log(
                f'Cache hit rate: {cache_hit_rate:.2f}% on cache size {len(self.inference_cache)} ({size_in_mb:.2f} MB)',
                level=LogLevel.DEBUG,
            )

    def clear_cache(self) -> None:
        self.inference_cache.clear()

    def add(self, hash: int, moves: MoveList, value: float) -> None:
        self.inference_cache[hash] = moves, value

    def get_encoded(self, hash: int) -> tuple[MoveList, float]:
        assert hash in self.inference_cache, f'Hash {hash} not in cache'
        moves, value = self.inference_cache[hash]

        return moves, value

    def get(self, hash: int) -> tuple[list[MoveScore], float]:
        moves, value = self.get_encoded(hash)

        return [(int(move), float(prob)) for move, prob in moves], value

    def __contains__(self, hash: int) -> bool:
        return hash in self.inference_cache
