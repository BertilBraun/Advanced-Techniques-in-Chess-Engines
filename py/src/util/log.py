from __future__ import annotations

import os
import time
from enum import Enum
from pprint import pprint
from typing import IO, Any


def datetime_str() -> str:
    return date_str() + ' ' + time_str()


def date_str() -> str:
    return time.strftime('%Y-%m-%d')


def time_str() -> str:
    return time.strftime('%H.%M.%S')


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


LOG_FOLDER = os.environ.get('TRAINING_LOG_PATH', 'logs')
LOG_LEVEL = LogLevel.INFO
_GLOBAL_LOG_FILE: IO[str] | None = None


def _global_log_file() -> IO[str]:
    global _GLOBAL_LOG_FILE
    if _GLOBAL_LOG_FILE is None:
        os.makedirs(LOG_FOLDER, exist_ok=True)
        # Append plus a per-process timestamped name: two processes in the same second must not truncate
        # each other's log.
        _GLOBAL_LOG_FILE = open(f'{LOG_FOLDER}/log_{date_str()}_{time_str()}_{os.getpid()}.log', 'a')
    return _GLOBAL_LOG_FILE


def log(*args: Any, level: LogLevel = LogLevel.INFO, use_pprint: bool = False, **kwargs: Any) -> None:
    timestamp = f'[{time_str()}]'
    log_level = f'[{level.name}]'
    log_file = _global_log_file()

    if use_pprint:
        print(timestamp, log_level, end=' ', file=log_file, flush=True)
        pprint(*args, **kwargs, stream=log_file, width=200)
        log_file.flush()
        if level.value >= LOG_LEVEL.value:
            pprint(*args, **kwargs, width=220)
    else:
        print(timestamp, log_level, *args, **kwargs, file=log_file, flush=True)
        if level.value >= LOG_LEVEL.value:
            print(timestamp, log_level, *args, **kwargs, flush=True)


def warn(*args: Any, use_pprint: bool = False, **kwargs: Any) -> None:
    log(*args, level=LogLevel.WARNING, use_pprint=use_pprint, **kwargs)


def error(*args: Any, use_pprint: bool = False, **kwargs: Any) -> None:
    log(*args, level=LogLevel.ERROR, use_pprint=use_pprint, **kwargs)
