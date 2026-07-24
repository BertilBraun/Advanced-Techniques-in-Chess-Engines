from pathlib import Path

import pytest

from src.util.communication import Communication


def test_heartbeat_uses_atomic_persistent_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communication = Communication(str(tmp_path))
    published_values: list[tuple[str, str]] = []

    def publish(identifier: str, value: str) -> None:
        published_values.append((identifier, value))

    monkeypatch.setattr(communication, 'publish_persistent_value', publish)

    communication.send_heartbeat('SELF PLAY 3')

    assert len(published_values) == 1
    identifier, timestamp = published_values[0]
    assert identifier == 'HEARTBEAT_SELF PLAY 3'
    assert float(timestamp) > 0.0
