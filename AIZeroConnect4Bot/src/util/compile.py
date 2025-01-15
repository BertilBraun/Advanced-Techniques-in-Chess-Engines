import torch
from typing import TypeVar

from src.util.log import log


T = TypeVar('T')


def try_compile(element: T) -> T:
    try:
        return torch.jit.script(element)  # type: ignore
    except:  # noqa
        log('Warning: Could not compile element.')

    return element
