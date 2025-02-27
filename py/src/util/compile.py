import torch
from typing import TypeVar


T = TypeVar('T')


def try_compile(element: T) -> T:
    from src.settings import USE_GPU

    if not USE_GPU:
        return element

    try:
        return torch.compile(element)  # type: ignore
    except:  # noqa
        print('Warning: Could not compile element:', element)

    try:
        return torch.jit.script(element)  # type: ignore
    except:  # noqa
        print('Warning: Could not jit element:', element)

    return element
