import time
import os
import torch
import numpy as np
import multiprocessing
from contextvars import ContextVar, Token
from pathlib import Path
from types import TracebackType
from tensorboardX import SummaryWriter
from tensorboardX.summary import scalar
from tensorboardX.writer import FileWriter
from typing import SupportsFloat


LOG_HISTOGRAMS = True  # Log any histograms to tensorboard - not sure, might be really slow, not sure though


class RestartSafeSummaryWriter(SummaryWriter):
    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        resolved_walltime = time.time() if walltime is None else walltime
        root_log_directory = self._get_file_writer().get_logdir()
        assert self.all_writers is not None
        for tag, scalar_value in tag_scalar_dict.items():
            child_log_directory = str(Path(root_log_directory) / main_tag / tag)
            child_writer = self.all_writers.get(child_log_directory)
            if child_writer is None:
                child_writer = FileWriter(
                    logdir=child_log_directory,
                    max_queue=self._max_queue,
                    flush_secs=self._flush_secs,
                    filename_suffix=self._filename_suffix,
                )
                self.all_writers[child_log_directory] = child_writer
            child_writer.add_summary(scalar(main_tag, scalar_value), global_step, resolved_walltime)


_TB_SUMMARY = ContextVar[RestartSafeSummaryWriter | None]('tensorboard_summary', default=None)
_TB_LOGGING_ENABLED = ContextVar[bool]('tensorboard_logging_enabled', default=True)
_HAS_PROMPTED = False
_TENSORBOARD_RUN_DIRECTORY_ENVIRONMENT_VARIABLE = 'TRAINING_TENSORBOARD_RUN_DIRECTORY'


def configure_tensorboard_run_directory(run_directory: str) -> None:
    allowed_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
    valid_initial_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    if (
        not run_directory
        or run_directory[0] not in valid_initial_characters
        or any(character not in allowed_characters for character in run_directory)
    ):
        raise ValueError('TensorBoard run directory must contain only letters, numbers, underscores, and hyphens.')
    os.environ[_TENSORBOARD_RUN_DIRECTORY_ENVIRONMENT_VARIABLE] = run_directory


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
        run_directory = os.environ.get(_TENSORBOARD_RUN_DIRECTORY_ENVIRONMENT_VARIABLE, f'run_{run}')
        self.log_folder = str(Path(log_root) / run_directory / suffix)
        if postfix_pid:
            self.log_folder += f'/{multiprocessing.current_process().pid}'
        self.enabled = enabled
        self._summary_token: Token[RestartSafeSummaryWriter | None] | None = None
        self._enabled_token: Token[bool] | None = None

    def __enter__(self) -> None:
        assert _TB_SUMMARY.get() is None, 'Only one tensorboard writer can be active at a time'
        self._enabled_token = _TB_LOGGING_ENABLED.set(self.enabled)
        if self.enabled:
            filename_suffix = f'.{time.time_ns()}_{multiprocessing.current_process().pid}'
            self._summary_token = _TB_SUMMARY.set(
                RestartSafeSummaryWriter(self.log_folder, filename_suffix=filename_suffix)
            )

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
