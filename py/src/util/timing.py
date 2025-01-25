import time

from functools import wraps


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
    from src.util.log import LogLevel, log
    from src.settings import log_scalar

    global function_times

    total_time = sum(function_times.values())
    for key, value in function_times.items():
        global_function_times[key] = global_function_times.get(key, 0.0) + value
    global_total_time = sum(global_function_times.values())

    if total_time > 0:
        id = int(time.time() - start_timing_time)
        for key in sorted(global_function_times.keys(), key=lambda x: global_function_times[x], reverse=True):
            log_scalar(
                f'function_percent_of_execution_time/{key}', global_function_times[key] / global_total_time * 100, id
            )
            log_scalar(f'function_total_time/{key}', global_function_times[key], id)
            log_scalar(f'function_total_invocations/{key}', global_function_invocations[key], id)
            log(
                f'{function_times.get(key, 0.0) / total_time:.2%} (total {global_function_times[key] / global_total_time:.2%} on {global_function_invocations[key]} invocations) {key}',
                level=LogLevel.DEBUG,
            )

        log_scalar(
            'function_time/total_traced_percent', global_total_time / (time.time() - start_timing_time) * 100, id
        )
        log(f'In total: {global_total_time / (time.time() - start_timing_time):.2%} recorded', level=LogLevel.DEBUG)

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
