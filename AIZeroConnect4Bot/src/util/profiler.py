import csv
import threading
import time
import psutil
import GPUtil
from datetime import datetime

from src.settings import USE_PROFILING


# Usage Logger
def _log_system_usage(interval: float):
    """
    Logs CPU, RAM, GPU, and VRAM usage every 'interval' seconds to a CSV file.
    """
    # Get the current process
    process = psutil.Process()

    with open('usage.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(
            [
                'timestamp',
                'rank',
                'cpu_percent',
                'ram_usage',
                'gpu_id',
                'gpu_load',
                'gpu_memory_used',
                'gpu_memory_total',
            ]
        )

        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for gpu in GPUtil.getGPUs():
                writer.writerow([timestamp, -1, 0, 0, gpu.id, gpu.load, gpu.memoryUsed, gpu.memoryTotal])

            for i, proc in enumerate([process] + process.children(recursive=True)):
                cpu_percent = proc.cpu_percent(interval=None)
                ram_usage = proc.memory_info().rss / 2**20
                writer.writerow([timestamp, i, cpu_percent, ram_usage, -1, 0, 0, 0])

            file.flush()  # Ensure data is written to disk
            time.sleep(interval)


def start_usage_logger():
    """
    Starts the system usage logger in a separate daemon thread.
    """
    if not USE_PROFILING:
        return

    logger_thread = threading.Thread(target=_log_system_usage, args=(1.0,), daemon=True)
    logger_thread.start()
