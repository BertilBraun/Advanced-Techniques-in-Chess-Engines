from contextlib import contextmanager
import traceback

from src.util.log import log


@contextmanager
def log_exceptions(name: str):
    try:
        yield
    except Exception as e:
        log(f'Exception in {name}: {e}')
        traceback.print_exc()
        exit()
