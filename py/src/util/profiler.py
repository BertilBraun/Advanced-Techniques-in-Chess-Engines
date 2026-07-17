import time
import psutil
import GPUtil
from threading import Event, Thread

from src.settings import log_scalar

from src.util.tensorboard import TensorboardWriter
from src.util.background_worker import BackgroundWorker


def _tensorboard_gpu_usage(run: int, interval: float, stop_event: Event) -> None:
    """Logs GPU usage every 'interval' seconds to tensorboard."""

    with TensorboardWriter(run, 'gpu_usage', postfix_pid=False):
        while not stop_event.is_set():
            for gpu in GPUtil.getGPUs():
                log_scalar(f'gpu/{gpu.id}/load', gpu.load, int(time.time() / interval))
                log_scalar(f'gpu/{gpu.id}/memory_used', gpu.memoryUsed, int(time.time() / interval))
            stop_event.wait(interval)


def _tensorboard_cpu_usage(
    run: int,
    interval: float,
    title: str,
    pid: int,
    stop_event: Event,
) -> None:
    """Logs CPU usage every 'interval' seconds to tensorboard."""

    with TensorboardWriter(run, f'cpu_usage_{title}', postfix_pid=True):
        process = psutil.Process(pid)
        while not stop_event.is_set():
            try:
                cpu_percent = process.cpu_percent(interval=None)
                ram_usage = process.memory_info().rss / 2**20
                log_scalar(f'cpu/{title}_{process.pid}/percent', cpu_percent, int(time.time() / interval))
                log_scalar(f'cpu/{title}_{process.pid}/ram_MB', ram_usage, int(time.time() / interval))
            except psutil.NoSuchProcess:
                break
            stop_event.wait(interval)


def start_gpu_usage_logger(run: int) -> BackgroundWorker:
    """Starts the GPU usage logger in a separate daemon thread."""

    stop_event = Event()
    thread = Thread(
        target=_tensorboard_gpu_usage,
        args=(run, 10.0, stop_event),
        daemon=True,
        name='gpu-usage-logger',
    )
    thread.start()
    return BackgroundWorker(thread=thread, stop_event=stop_event)


def start_cpu_usage_logger(run: int, title: str) -> BackgroundWorker:
    """Starts the CPU usage logger in a separate daemon thread."""

    stop_event = Event()
    thread = Thread(
        target=_tensorboard_cpu_usage,
        args=(run, 10.0, title, psutil.Process().pid, stop_event),
        daemon=True,
        name=f'{title}-usage-logger',
    )
    thread.start()
    return BackgroundWorker(thread=thread, stop_event=stop_event)
