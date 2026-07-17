from dataclasses import dataclass
from threading import Event, Thread


@dataclass(frozen=True)
class BackgroundWorker:
    thread: Thread
    stop_event: Event

    def stop(self, timeout_seconds: float = 10.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout_seconds)
        if self.thread.is_alive():
            raise RuntimeError(f'Background worker {self.thread.name!r} did not stop.')
