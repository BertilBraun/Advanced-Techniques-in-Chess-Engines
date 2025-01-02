import torch
from typing import TypeVar


T = TypeVar('T')


def try_compile(element: T) -> T:
    try:
        return torch.compile(element)  # type: ignore
    except:  # noqa
        pass
    try:
        return torch.jit.script(element)  # type: ignore
    except:  # noqa
        pass

    return element
