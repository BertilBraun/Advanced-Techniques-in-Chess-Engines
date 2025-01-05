import csv
import threading
import time
import psutil
import GPUtil
from datetime import datetime

from src.settings import USE_PROFILING


# Usage Logger
def _log_system_usage(interval: float, rank: int):
    """
    Logs CPU, RAM, GPU, and VRAM usage every 'interval' seconds to a CSV file.
    """
    # Get the current process
    process = psutil.Process()

    with open(f'usage_{rank}.csv', mode='w', newline='') as file:
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
            cpu_percent = process.cpu_percent(interval=None)
            ram_usage = process.memory_info().rss / 2**20  # Convert to MB

            gpus: list[GPUtil.GPU] = GPUtil.getGPUs()
            if gpus:
                gpu_id = rank % len(gpus)
                gpu_load = gpus[gpu_id].load * 100  # Percentage
                gpu_memory_used = gpus[gpu_id].memoryUsed
                gpu_memory_total = gpus[gpu_id].memoryTotal
            else:
                gpu_id = -1
                gpu_load = 0
                gpu_memory_used = 0
                gpu_memory_total = 0

            writer.writerow(
                [timestamp, rank, cpu_percent, ram_usage, gpu_id, gpu_load, gpu_memory_used, gpu_memory_total]
            )
            file.flush()  # Ensure data is written to disk
            time.sleep(interval)


def start_usage_logger(rank: int):
    """
    Starts the system usage logger in a separate daemon thread.
    """
    if not USE_PROFILING:
        return

    logger_thread = threading.Thread(target=_log_system_usage, args=(1.0, rank), daemon=True)
    logger_thread.start()
