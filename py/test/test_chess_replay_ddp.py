from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
import sys
import time
from types import ModuleType

sys.modules.setdefault('GPUtil', ModuleType('GPUtil'))

from src.cluster.TrainerProcess import available_tcp_port
from test_helpers.chess_replay_ddp import WORLD_SIZE, run_replay_rank


def test_gloo_ranks_receive_identical_snapshot_and_disjoint_samples() -> None:
    context = multiprocessing.get_context('spawn')
    initialization_method = f'tcp://127.0.0.1:{available_tcp_port()}'
    parents: list[Connection] = []
    processes: list[multiprocessing.Process] = []
    for rank in range(WORLD_SIZE):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=run_replay_rank, args=(rank, initialization_method, child))
        process.start()
        child.close()
        parents.append(parent)
        processes.append(process)
    deadline = time.monotonic() + 30
    while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
        time.sleep(0.05)
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        assert process.exitcode == 0
    results = tuple(parent.recv() for parent in parents)
    for parent in parents:
        parent.close()

    assert {result[:4] for result in results} == {(12, 8, 4, 1_024)}
    assert not set(results[0][4]) & set(results[1][4])
