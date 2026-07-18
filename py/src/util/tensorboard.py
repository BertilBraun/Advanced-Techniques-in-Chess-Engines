import time
import os
import torch
import numpy as np
import multiprocessing
from contextvars import ContextVar, Token
from types import TracebackType
from tensorboardX import SummaryWriter
from typing import SupportsFloat


LOG_HISTOGRAMS = True  # Log any histograms to tensorboard - not sure, might be really slow, not sure though

_TB_SUMMARY = ContextVar[SummaryWriter | None]('tensorboard_summary', default=None)
_TB_LOGGING_ENABLED = ContextVar[bool]('tensorboard_logging_enabled', default=True)
_HAS_PROMPTED = False


def _tb_check_active() -> bool:
    if not _TB_LOGGING_ENABLED.get():
        return False
    if _TB_SUMMARY.get() is None:
        global _HAS_PROMPTED
        if not _HAS_PROMPTED:
            from src.util.log import LogLevel, log

            log('Warning: No tensorboard writer active', level=LogLevel.WARNING)
            _HAS_PROMPTED = True
        return False
    return True


def log_scalar(name: str, value: float, iteration: int | None = None) -> None:
    if not _tb_check_active():
        return
    summary = _TB_SUMMARY.get()
    assert summary is not None, 'No tensorboard writer active'
    if iteration is None:
        iteration = int(time.time() * 1000)
    summary.add_scalar(name, value, iteration)


def log_scalars(name: str, values: dict[str, SupportsFloat], iteration: int | None = None) -> None:
    if not _tb_check_active() or not values:
        return
    summary = _TB_SUMMARY.get()
    assert summary is not None, 'No tensorboard writer active'
    if iteration is None:
        iteration = int(time.time() * 1000)
    values_float = {k: float(v) for k, v in values.items()}  # Ensure all values are floats
    summary.add_scalars(name, values_float, iteration)


def log_text(name: str, text: str, iteration: int | None = None) -> None:
    if not _tb_check_active():
        return
    summary = _TB_SUMMARY.get()
    assert summary is not None, 'No tensorboard writer active'
    if iteration is None:
        iteration = int(time.time() * 1000)
    summary.add_text(name, text, iteration)


def log_histogram(name: str, values: torch.Tensor | np.ndarray, iteration: int | None = None) -> None:
    if not LOG_HISTOGRAMS or not _tb_check_active():
        return
    values = values.reshape(-1)
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    if not values.size:
        return
    summary = _TB_SUMMARY.get()
    assert summary is not None, 'No tensorboard writer active'
    if iteration is None:
        iteration = int(time.time() * 1000)
    summary.add_histogram(name, values, iteration)


class TensorboardWriter:
    def __init__(
        self,
        run: int,
        suffix: str = '',
        postfix_pid: bool = True,
        enabled: bool = True,
    ) -> None:
        from src.settings import LOG_FOLDER

        log_root = os.environ.get('TRAINING_TENSORBOARD_LOG_PATH', LOG_FOLDER)
        self.log_folder = f'{log_root}/run_{run}/{suffix}'
        if postfix_pid:
            self.log_folder += f'/{multiprocessing.current_process().pid}'
        self.enabled = enabled
        self._summary_token: Token[SummaryWriter | None] | None = None
        self._enabled_token: Token[bool] | None = None

    def __enter__(self) -> None:
        assert _TB_SUMMARY.get() is None, 'Only one tensorboard writer can be active at a time'
        self._enabled_token = _TB_LOGGING_ENABLED.set(self.enabled)
        if self.enabled:
            self._summary_token = _TB_SUMMARY.set(SummaryWriter(self.log_folder))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._enabled_token is not None, 'Tensorboard writer context was not entered'
        if self.enabled:
            summary = _TB_SUMMARY.get()
            assert summary is not None, 'No tensorboard writer active'
            assert self._summary_token is not None, 'Tensorboard writer context was not entered'
            summary.close()
            _TB_SUMMARY.reset(self._summary_token)
            self._summary_token = None
        _TB_LOGGING_ENABLED.reset(self._enabled_token)
        self._enabled_token = None
