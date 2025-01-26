import time

import asyncio

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
    global function_times

    total_time = sum(function_times.values())
    for key, value in function_times.items():
        global_function_times[key] = global_function_times.get(key, 0.0) + value
    global_total_time = sum(global_function_times.values())

    for key in sorted(global_function_times.keys(), key=lambda x: global_function_times[x], reverse=True):
        print(
            f'{function_times.get(key, 0.0) / total_time:.2%} (total {global_function_times[key] / global_total_time:.2%} on {global_function_invocations[key]} invocations) {key}',
        )

    print(f'total time spent: {global_total_time:.5f}, total time: {time.time() - start_timing_time:.5f}')
    print(f'total traced time: { global_total_time / (time.time() - start_timing_time):.2%}')
    res()


def res():
    global function_times, global_function_times, global_function_invocations, start_timing_time, call_stack
    function_times = {}
    global_function_times = {}
    global_function_invocations = {}
    start_timing_time = -1
    call_stack = []


@timeit
def sync_recursive_work(level, n):
    """Perform synchronous recursive Fibonacci calculations."""
    if level == 0:
        return fibonacci(n)
    sync_recursive_work(level - 1, n)


@timeit
def fibonacci(n):
    """Compute Fibonacci numbers."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@timeit
async def async_recursive_work(level, n):
    """Perform asynchronous recursive Fibonacci calculations."""
    if level == 0:
        return await async_fibonacci(n)
    return await async_recursive_work(level - 1, n)


@timeit
async def async_fibonacci(n):
    """Asynchronous Fibonacci computation."""
    if n <= 1:
        return n
    return sum(
        await asyncio.gather(
            async_fibonacci(n - 1),
            async_fibonacci(n - 2),
        )
    )


async def main():
    global start_timing_time
    start_timing_time = time.time()

    fib_input = 20  # Fibonacci input size
    recursion_depth = 3  # Levels of recursion

    iterations = 30

    print('=' * 80)
    print('Iteration - Synchronous Recursive Work')
    for iteration in range(iterations):
        sync_recursive_work(recursion_depth, fib_input)
    reset_times()

    print('=' * 80)
    print('Iteration - Asynchronous Recursive Work')
    for iteration in range(iterations):
        await async_recursive_work(recursion_depth, fib_input)
    reset_times()

    print('Done!')


# Run the loop
if __name__ == '__main__':
    asyncio.run(main())
