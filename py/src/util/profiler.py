import csv
import threading
import time
import psutil
import GPUtil
from datetime import datetime

from src.settings import USE_PROFILING, log_scalar, log_scalars
from functools import wraps

from src.util.log import LogLevel, log


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
                try:
                    cpu_percent = proc.cpu_percent(interval=None)
                    ram_usage = proc.memory_info().rss / 2**20
                    writer.writerow([timestamp, i, cpu_percent, ram_usage, -1, 0, 0, 0])
                except psutil.NoSuchProcess:
                    pass

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


# Global variables to accumulate times
function_times = {}
global_function_times = {}
global_function_invocations = {}
start_timing_time = -1
call_stack = []


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global call_stack, function_times, start_timing_time, global_function_invocations
        start_time = time.time()

        if start_timing_time == -1:
            start_timing_time = start_time

        call_stack.append(func.__name__)
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            elapsed = end_time - start_time
            call_stack.pop()
            # Accumulate time
            function_times[func.__name__] = function_times.get(func.__name__, 0.0) + elapsed
            global_function_invocations[func.__name__] = global_function_invocations.get(func.__name__, 0) + 1
            # Subtract from parent function
            if call_stack:
                parent = call_stack[-1]
                function_times[parent] = function_times.get(parent, 0.0) - elapsed

    return wrapper


def reset_times():
    global function_times
    total_time = sum(function_times.values())
    for key, value in function_times.items():
        global_function_times[key] = global_function_times.get(key, 0.0) + value
    global_total_time = sum(global_function_times.values())

    if total_time > 0:
        id = int((time.time() - start_timing_time) * 1000)
        for key in sorted(global_function_times.keys(), key=lambda x: global_function_times[x], reverse=True):
            log_scalars(
                f'function_time/{key}',
                {
                    'time': global_function_times[key] / global_total_time,
                    'invocations': global_function_invocations[key] / global_total_time,
                },
                id,
            )
            log(
                f'{function_times.get(key, 0.0) / total_time:.2%} (total {global_function_times[key] / global_total_time:.2%} on {global_function_invocations[key]} invocations) {key}',
                level=LogLevel.DEBUG,
            )

        log_scalar('function_time/total', global_total_time / (time.time() - start_timing_time), id)
        log(f'In total: {global_total_time / (time.time() - start_timing_time):.2%} recorded', level=LogLevel.DEBUG)

        log(f'In total: {global_total_time / (time.time() - start_timing_time):.2%} recorded')

    function_times = {}


if __name__ == '__main__':

    @timeit
    def test1():
        time.sleep(1)

    @timeit
    def test2():
        time.sleep(1)
        test1()
        time.sleep(1)

    for _ in range(2):
        test2()

    print(function_times)
