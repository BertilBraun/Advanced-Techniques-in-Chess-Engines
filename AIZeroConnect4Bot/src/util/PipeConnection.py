from typing import Any, Protocol
# import readable buffer


class PipeConnection(Protocol):
    # Define the pipe connection interface from the multiprocessing module just as a type definition.
    @property
    def readable(self) -> bool:
        ...

    @property
    def writable(self) -> bool:
        ...

    def send_bytes(self, buf, offset: int = 0, size: int | None = None) -> None:
        ...

    def recv_bytes(self) -> bytes:
        ...

    def poll(self, timeout: float = 0.0) -> bool:
        ...

    def recv(self) -> str:
        ...

    def send(self, obj: Any) -> None:
        ...

    def close(self) -> None:
        ...
