import threading
import time
import psutil
import GPUtil
from datetime import datetime
from contextlib import contextmanager
import csv

from src.settings import USE_PROFILING

SYSTEM_USAGE_FILENAME = 'system_usage.csv'
EVENTS_FILENAME = 'events.csv'


# Usage Logger
def _log_system_usage(interval=1):
    """
    Logs CPU, RAM, GPU, and VRAM usage every 'interval' seconds to a CSV file.
    """
    open(EVENTS_FILENAME, 'w').close()  # Clear events file

    # Get the current process
    process = psutil.Process()

    with open(SYSTEM_USAGE_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['timestamp', 'cpu_percent', 'ram_usage', 'gpu_load', 'gpu_memory_used', 'gpu_memory_total'])

        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cpu_percent = process.cpu_percent(interval=None)
            ram_usage = process.memory_info().rss / 2**20  # Convert to MB

            gpus: list[GPUtil.GPU] = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100  # Percentage
                gpu_memory_used = gpus[0].memoryUsed
                gpu_memory_total = gpus[0].memoryTotal
            else:
                gpu_load = 0
                gpu_memory_used = 0
                gpu_memory_total = 0

            writer.writerow([timestamp, cpu_percent, ram_usage, gpu_load, gpu_memory_used, gpu_memory_total])
            file.flush()  # Ensure data is written to disk
            time.sleep(interval)


def start_usage_logger():
    """
    Starts the system usage logger in a separate daemon thread.
    """
    if not USE_PROFILING:
        return
    logger_thread = threading.Thread(target=_log_system_usage, daemon=True)
    logger_thread.start()


# Event Logger as Context Manager
@contextmanager
def log_event(event_name):
    """
    Context manager to log the start and end of an event.
    """

    def _log_event(start_end):
        if not USE_PROFILING:
            return
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(EVENTS_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, start_end, event_name])
            file.flush()

    _log_event('START')
    time.sleep(2)
    try:
        yield
    finally:
        time.sleep(2)
        _log_event('END')


if __name__ == '__main__':
    # Simulated Self-Play and Training Functions
    def self_play():
        with log_event('self_play'):
            print('Self-play started.')
            time.sleep(5)  # Simulate self-play duration
            print('Self-play ended.')

    def training():
        with log_event('training'):
            print('Training started.')
            time.sleep(10)  # Simulate training duration
            print('Training ended.')

    start_usage_logger()
    self_play()
    training()
    print('All tasks completed.')
