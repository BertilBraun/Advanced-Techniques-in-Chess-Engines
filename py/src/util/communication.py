# defines a class which allows for communication between the main thread and the worker threads
# for now this is via files, but in the future this could be via sockets or other means
from __future__ import annotations

from pathlib import Path


class Communication:
    def __init__(self, folder: str) -> None:
        self.folder = Path(folder)
        if not self.folder.exists():
            self.folder.mkdir(parents=True, exist_ok=True)

    def boardcast(self, identifier: str, message: str = '') -> None:
        """Sends a message by creating a file with the identifier."""
        with open(self._file_path(identifier), 'w') as f:
            f.write(message)

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

    def _file_path(self, identifier: str) -> Path:
        """Returns the file path for a given identifier."""
        return self.folder / f'{identifier}.txt'
