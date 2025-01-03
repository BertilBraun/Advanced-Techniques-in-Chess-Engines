from contextlib import contextmanager

from src.util.log import log


@contextmanager
def log_exceptions(name: str):
    try:
        yield
    except Exception as e:
        log(f'Exception in {name}: {e}')
