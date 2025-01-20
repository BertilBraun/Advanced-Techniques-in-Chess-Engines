import json
import os
import random
import time
from typing import TypeVar


T = TypeVar('T')


def lerp(a: T, b: T, t: float) -> T:
    return (1 - t) * a + t * b  # type: ignore


def random_id(seed_new: bool = True) -> str:
    if seed_new:
        random.seed(time.time())
    random_base = 1_000_000_000
    return str(random.randint(random_base, random_base * 10))


def load_json(path: str | os.PathLike) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
