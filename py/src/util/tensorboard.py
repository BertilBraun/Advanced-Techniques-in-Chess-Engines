import torch
import numpy as np
import multiprocessing
from contextlib import contextmanager
from tensorboardX import SummaryWriter

LOG_HISTOGRAMS = True  # Log any histograms to tensorboard - not sure, might be really slow, not sure though

_TB_SUMMARY: SummaryWriter | None = None


def log_scalar(name: str, value: float, iteration: int) -> None:
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    _TB_SUMMARY.add_scalar(name, value, iteration)


def log_scalars(name: str, values: dict[str, float], iteration: int) -> None:
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    _TB_SUMMARY.add_scalars(name, values, iteration)


def log_text(name: str, text: str, iteration: int | None = None) -> None:
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    _TB_SUMMARY.add_text(name, text, iteration)


def log_histogram(name: str, values: torch.Tensor | np.ndarray, iteration: int) -> None:
    if not LOG_HISTOGRAMS:
        return
    values = values.reshape(-1)
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    assert _TB_SUMMARY is not None, 'No tensorboard writer active'
    _TB_SUMMARY.add_histogram(name, values, iteration)


@contextmanager
def tensorboard_writer():
    from src.settings import LOG_FOLDER

    global _TB_SUMMARY
    assert _TB_SUMMARY is None, 'Only one tensorboard writer can be active at a time'
    _TB_SUMMARY = SummaryWriter(LOG_FOLDER + f'/{multiprocessing.current_process().pid}')
    try:
        yield
    finally:
        _TB_SUMMARY.close()
        _TB_SUMMARY = None
