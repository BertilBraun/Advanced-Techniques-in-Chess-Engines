import json
import os
import random
import time
from typing import TypeVar


T = TypeVar('T')


def lerp(a: T, b: T, t: float) -> T:
    return (1 - t) * a + t * b  # type: ignore


def clamp(value: T, min_value: T, max_value: T) -> T:
    assert isinstance(value, (int, float)), 'Value must be an int or float'
    assert isinstance(min_value, (int, float)), 'Min value must be an int or float'
    assert isinstance(max_value, (int, float)), 'Max value must be an int or float'
    assert min_value <= max_value, 'Min value must be less than or equal to max value'
    return max(min(value, max_value), min_value)


def random_id(seed_new: bool = True) -> str:
    if seed_new:
        random.seed(time.time())
    random_base = 1_000_000_000
    return str(random.randint(random_base, random_base * 10))


def load_json(path: str | os.PathLike) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
