import time
import psutil
import GPUtil
from multiprocessing import Process

from src.settings import log_scalar

from src.util.tensorboard import TensorboardWriter


def _tensorboard_gpu_usage(run: int, interval: float):
    """Logs GPU usage every 'interval' seconds to tensorboard."""

    with TensorboardWriter(run, 'gpu_usage', postfix_pid=False):
        while True:
            for gpu in GPUtil.getGPUs():
                log_scalar(f'gpu/{gpu.id}/load', gpu.load, int(time.time() / interval))
                log_scalar(f'gpu/{gpu.id}/memory_used', gpu.memoryUsed, int(time.time() / interval))
            time.sleep(interval)


def _tensorboard_cpu_usage(run: int, interval: float, title: str, pid: int):
    """Logs CPU usage every 'interval' seconds to tensorboard."""

    with TensorboardWriter(run, f'cpu_usage_{title}', postfix_pid=True):
        process = psutil.Process(pid)
        while True:
            try:
                cpu_percent = process.cpu_percent(interval=None)
                ram_usage = process.memory_info().rss / 2**20
                log_scalar(f'cpu/{title}_{process.pid}/percent', cpu_percent, int(time.time() / interval))
                log_scalar(f'cpu/{title}_{process.pid}/ram_MB', ram_usage, int(time.time() / interval))
            except psutil.NoSuchProcess:
                break
            time.sleep(interval)


def start_gpu_usage_logger(run: int):
    """Starts the GPU usage logger in a separate daemon thread."""

    Process(
        target=_tensorboard_gpu_usage,
        args=(run, 10.0),
        daemon=True,
    ).start()


def start_cpu_usage_logger(run: int, title: str, pid: int):
    """Starts the CPU usage logger in a separate daemon thread."""

    Process(
        target=_tensorboard_cpu_usage,
        args=(run, 10.0, title, pid),
        daemon=True,
    ).start()
