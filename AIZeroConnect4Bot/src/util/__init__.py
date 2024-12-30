import json
import os
import random
from typing import Iterable, TypeVar


T = TypeVar('T')


def lerp(a: T, b: T, t: float) -> T:
    return (1 - t) * a + t * b  # type: ignore


def random_id() -> str:
    random_base = 1_000_000_000
    return str(random.randint(random_base, random_base * 10))


def batched_iterate(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_json(path: str | os.PathLike) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
