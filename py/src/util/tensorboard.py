import time
import torch
import numpy as np
import multiprocessing
from tensorboardX import SummaryWriter
from typing import SupportsFloat


LOG_HISTOGRAMS = True  # Log any histograms to tensorboard - not sure, might be really slow, not sure though

_TB_SUMMARY: SummaryWriter | None = None
_HAS_PROMPTED = False


def _tb_check_active() -> bool:
    if _TB_SUMMARY is None:
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
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    if iteration is None:
        iteration = int(time.time() * 1000)
    _TB_SUMMARY.add_scalar(name, value, iteration)


def log_scalars(name: str, values: dict[str, SupportsFloat], iteration: int | None = None) -> None:
    if not _tb_check_active():
        return
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    if iteration is None:
        iteration = int(time.time() * 1000)
    values_float = {k: float(v) for k, v in values.items()}  # Ensure all values are floats
    _TB_SUMMARY.add_scalars(name, values_float, iteration)


def log_text(name: str, text: str, iteration: int | None = None) -> None:
    if not _tb_check_active():
        return
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    if iteration is None:
        iteration = int(time.time() * 1000)
    _TB_SUMMARY.add_text(name, text, iteration)


def log_histogram(name: str, values: torch.Tensor | np.ndarray, iteration: int | None = None) -> None:
    if not LOG_HISTOGRAMS or not _tb_check_active():
        return
    values = values.reshape(-1)
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    if iteration is None:
        iteration = int(time.time() * 1000)
    _TB_SUMMARY.add_histogram(name, values, iteration)


class TensorboardWriter:
    def __init__(self, run: int, suffix: str = '', postfix_pid: bool = True) -> None:
        from src.settings import LOG_FOLDER

        self.log_folder = f'{LOG_FOLDER}/run_{run}/{suffix}'
        if postfix_pid:
            self.log_folder += f'/{multiprocessing.current_process().pid}'

    def __enter__(self):
        global _TB_SUMMARY
        assert _TB_SUMMARY is None, 'Only one tensorboard writer can be active at a time'
        _TB_SUMMARY = SummaryWriter(self.log_folder)

    def __exit__(self, exc_type, exc_value, traceback):
        global _TB_SUMMARY
        assert _TB_SUMMARY is not None, 'No tensorboard writer active'
        _TB_SUMMARY.close()
        _TB_SUMMARY = None
