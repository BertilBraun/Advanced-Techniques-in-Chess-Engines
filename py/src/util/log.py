import os
import time

from enum import Enum
from pprint import pprint

from src.settings import LOG_FOLDER


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


LOG_FILE = LOG_FOLDER + f'/log_{date_str()}_{time_str()}.log'
LOG_LEVEL = LogLevel.INFO
os.makedirs(LOG_FOLDER, exist_ok=True)
GLOBAL_LOG_FILE = None


def log(
    *args,
    level: LogLevel = LogLevel.INFO,
    use_pprint: bool = False,
    log_file_name: str = LOG_FILE,
    **kwargs,
) -> None:
    timestamp = f'[{time_str()}]'
    log_level = f'[{level.name}]'

    if log_file_name != LOG_FILE:
        # ensure that the log file folder exists
        dir_name = os.path.dirname(log_file_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        log_file = open(log_file_name, 'a')
    else:
        global GLOBAL_LOG_FILE
        if GLOBAL_LOG_FILE is None:
            GLOBAL_LOG_FILE = open(LOG_FILE, 'w')
        log_file = GLOBAL_LOG_FILE

    if use_pprint:
        print(timestamp, log_level, end=' ', file=log_file, flush=True)
        pprint(*args, **kwargs, stream=log_file, width=200)
        log_file.flush()
        if level.value >= LOG_LEVEL.value:
            # print(timestamp, log_level, end=' ', flush=True)
            pprint(*args, **kwargs, width=220)
    else:
        print(timestamp, log_level, *args, **kwargs, file=log_file, flush=True)
        if level.value >= LOG_LEVEL.value:
            print(timestamp, log_level, *args, **kwargs, flush=True)

    if log_file_name != LOG_FILE:
        log_file.close()


def ratio(a: int, b: int) -> str:
    return f'{a}/{b} ({(a / b) * 100:.2f}%)'
