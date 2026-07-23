# defines a class which allows for communication between the main thread and the worker threads
# for now this is via files, but in the future this could be via sockets or other means
from __future__ import annotations

import os
from pathlib import Path
import time
import uuid


STOP_SELF_PLAY = 'STOP SELF PLAY'
SELF_PLAY_PAUSED = 'SELF PLAY PAUSED'
RESUME_SELF_PLAY = 'RESUME SELF PLAY'
SELF_PLAY_RESUMED = 'SELF PLAY RESUMED'
FLUSH_REPLAY_SHARD = 'FLUSH REPLAY SHARD'
SNAPSHOT_SELF_PLAY_STATISTICS = 'SNAPSHOT SELF PLAY STATISTICS'
START_CONTINUOUS_SELF_PLAY = 'START CONTINUOUS SELF PLAY'
LATEST_SELF_PLAY_MODEL_VERSION = 'LATEST SELF PLAY MODEL VERSION'


def refresh_self_play_model_message(model_version: int) -> str:
    if model_version < 0:
        raise ValueError('Model version must be nonnegative.')
    return f'REFRESH SELF PLAY MODEL: {model_version}'


def self_play_model_refreshed_message(model_version: int) -> str:
    if model_version < 0:
        raise ValueError('Model version must be nonnegative.')
    return f'SELF PLAY MODEL REFRESHED: {model_version}'


def update_self_play_search_schedule_message(schedule_version: int) -> str:
    if schedule_version < 0:
        raise ValueError('Search schedule version must be nonnegative.')
    return f'UPDATE SELF PLAY SEARCH SCHEDULE: {schedule_version}'


class Communication:
    def __init__(self, folder: str) -> None:
        self.folder = Path(folder)
        if not self.folder.exists():
            self.folder.mkdir(parents=True, exist_ok=True)

    def boardcast(self, identifier: str, message: str = '') -> None:
        """Sends a message by creating a file with the identifier."""
        with open(self._file_path(identifier), 'w') as f:
            f.write(message)

    def publish_persistent_value(self, identifier: str, value: str) -> None:
        """Atomically replace a persistent command value read by many workers."""
        path = self._file_path(identifier)
        temporary_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        with temporary_path.open('x', encoding='utf-8') as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)

    def is_received(self, identifier: str) -> bool:
        """Checks if a message has been received by checking for the existence of the file."""
        return self._file_path(identifier).exists()

    def receive(self, identifier: str) -> str:
        """Receives a message by reading the content of the file."""
        file_path = self._file_path(identifier)
        if not file_path.exists():
            raise FileNotFoundError(f'Message with identifier {identifier} not found.')

        with open(file_path, 'r') as f:
            content = f.read()

        file_path.unlink()
        return content

    def try_read(self, identifier: str) -> str | None:
        """Tries to read a message without raising an error if the file does not exist."""
        file_path = self._file_path(identifier)
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return f.read()

    def send_to_id(self, identifier: str, node_id: int) -> None:
        """Sends a message to a specific node by creating a file with the identifier and node ID."""
        file_path = self.folder / f'{identifier}_node_{node_id}.txt'
        with open(file_path, 'w') as f:
            f.write('')

    def try_receive_from_id(self, identifier: str, node_id: int) -> bool:
        """Tries to receive a message from a specific node by checking for the existence of the file."""
        file_path = self.folder / f'{identifier}_node_{node_id}.txt'
        if not file_path.exists():
            return False

        file_path.unlink()
        return True

    def clear_all(self) -> None:
        """Clears all messages by removing all files in the folder."""
        for file in self.folder.glob('*.txt'):
            file.unlink()

    def send_heartbeat(self, identifier: str) -> None:
        """Sends a heartbeat message by creating a file with the identifier."""
        self.boardcast(f'HEARTBEAT_{identifier}', str(time.time()))

    def is_alive(self, identifier: str, timeout: float) -> bool:
        """Checks if a heartbeat message has been received for the given identifier."""
        content = self.try_read(f'HEARTBEAT_{identifier}')
        if content is None:
            return True

        return float(content) > time.time() - timeout

    def _file_path(self, identifier: str) -> Path:
        """Returns the file path for a given identifier."""
        return self.folder / f'{identifier}.txt'


def pause_self_play_workers(
    communication: Communication,
    node_ids: tuple[int, ...],
    timeout_seconds: float,
) -> None:
    for node_id in node_ids:
        communication.try_receive_from_id(SELF_PLAY_PAUSED, node_id)
        communication.send_to_id(STOP_SELF_PLAY, node_id)

    pending_node_ids = set(node_ids)
    deadline = time.monotonic() + timeout_seconds
    while pending_node_ids:
        pending_node_ids = {
            node_id for node_id in pending_node_ids if not communication.try_receive_from_id(SELF_PLAY_PAUSED, node_id)
        }
        if not pending_node_ids:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Self-play workers did not pause before training: {sorted(pending_node_ids)}')
        time.sleep(0.05)


def resume_self_play_workers(
    communication: Communication,
    node_ids: tuple[int, ...],
    timeout_seconds: float,
) -> None:
    for node_id in node_ids:
        communication.try_receive_from_id(SELF_PLAY_RESUMED, node_id)
        communication.send_to_id(RESUME_SELF_PLAY, node_id)

    pending_node_ids = set(node_ids)
    deadline = time.monotonic() + timeout_seconds
    while pending_node_ids:
        pending_node_ids = {
            node_id for node_id in pending_node_ids if not communication.try_receive_from_id(SELF_PLAY_RESUMED, node_id)
        }
        if not pending_node_ids:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Self-play workers did not resume after training: {sorted(pending_node_ids)}')
        time.sleep(0.05)
