from contextlib import contextmanager
import traceback
from collections.abc import Iterator

from src.util.log import warn


@contextmanager
def log_exceptions(name: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        warn(f'Exception in {name}: {e}')
        traceback.print_exc()
        raise
