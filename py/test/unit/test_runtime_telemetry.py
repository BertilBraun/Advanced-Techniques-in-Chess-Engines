from __future__ import annotations

import struct
from pathlib import Path

from src.az.runtime.telemetry_journal import TelemetryJournal


def test_telemetry_journal_repairs_torn_final_frame(tmp_path: Path) -> None:
    path = (tmp_path / 'telemetry.azt').resolve()
    journal = TelemetryJournal(path)
    journal.append((b'first', b'second'))
    with path.open('ab') as stream:
        stream.write(struct.pack('<I', 20))
        stream.write(b'torn')

    recovered = TelemetryJournal(path)

    assert recovered.read_payloads() == (b'first', b'second')
    recovered.append((b'third',))
    assert recovered.read_payloads() == (b'first', b'second', b'third')
