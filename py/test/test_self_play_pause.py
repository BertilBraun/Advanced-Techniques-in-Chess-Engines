from pathlib import Path
from threading import Thread
from time import sleep

import pytest

from src.util.communication import (
    Communication,
    SELF_PLAY_PAUSED,
    STOP_SELF_PLAY,
    pause_self_play_workers,
)


def acknowledge_pause(communication: Communication, node_id: int) -> None:
    while not communication.try_receive_from_id(STOP_SELF_PLAY, node_id):
        sleep(0.01)
    communication.send_to_id(SELF_PLAY_PAUSED, node_id)


def test_pause_self_play_workers_waits_for_every_acknowledgement(tmp_path: Path) -> None:
    communication = Communication(str(tmp_path))
    node_ids = (2, 4)
    threads = tuple(
        Thread(target=acknowledge_pause, args=(communication, node_id), daemon=True) for node_id in node_ids
    )
    for thread in threads:
        thread.start()

    pause_self_play_workers(communication, node_ids, timeout_seconds=1)

    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()


def test_pause_self_play_workers_times_out_without_acknowledgement(tmp_path: Path) -> None:
    communication = Communication(str(tmp_path))

    with pytest.raises(RuntimeError, match=r'\[7\]'):
        pause_self_play_workers(communication, (7,), timeout_seconds=0.05)
