import torch
import numpy as np
import multiprocessing
from tensorboardX import SummaryWriter


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


def log_scalar(name: str, value: float, iteration: int) -> None:
    if not _tb_check_active():
        return
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    _TB_SUMMARY.add_scalar(name, value, iteration)


def log_scalars(name: str, values: dict[str, float | int], iteration: int) -> None:
    if not _tb_check_active():
        return
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    _TB_SUMMARY.add_scalars(name, values, iteration)


def log_text(name: str, text: str, iteration: int | None = None) -> None:
    if not _tb_check_active():
        return
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    _TB_SUMMARY.add_text(name, text, iteration)


def log_histogram(name: str, values: torch.Tensor | np.ndarray, iteration: int) -> None:
    if not LOG_HISTOGRAMS or not _tb_check_active():
        return
    values = values.reshape(-1)
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    _TB_SUMMARY.add_histogram(name, values, iteration)


class TensorboardWriter:
    def __init__(self, run: int, suffix: str = '', postfix_pid: bool = True) -> None:
        self.suffix = f'{run}/{suffix}_{multiprocessing.current_process().pid}' if postfix_pid else suffix

    def __enter__(self):
        from src.settings import LOG_FOLDER

        global _TB_SUMMARY
        assert _TB_SUMMARY is None, 'Only one tensorboard writer can be active at a time'
        _TB_SUMMARY = SummaryWriter(f'{LOG_FOLDER}/{self.suffix}')

    def __exit__(self, exc_type, exc_value, traceback):
        global _TB_SUMMARY
        assert _TB_SUMMARY is not None, 'No tensorboard writer active'
        _TB_SUMMARY.close()
        _TB_SUMMARY = None
