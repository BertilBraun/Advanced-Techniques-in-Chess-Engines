import json
import os
import random
import time
from typing import TypeVar


T = TypeVar('T')


def lerp(a: T, b: T, t: float) -> T:
    assert 0 <= t <= 1, f'Interpolation factor {t=} must be in the range [0, 1]'
    return (1 - t) * a + t * b  # type: ignore


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


def random_id(seed_new: bool = False) -> str:
    if seed_new:
        random.seed(time.time())
    random_base = 1_000_000_000
    return str(random.randint(random_base, random_base * 10))


def load_json(path: str | os.PathLike) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
