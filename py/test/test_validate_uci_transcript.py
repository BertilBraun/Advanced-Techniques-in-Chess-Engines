from pathlib import Path

import pytest

from tools.validate_uci_transcript import ValidationRequest, validate_transcript


def test_validate_transcript_accepts_legal_bestmove(tmp_path: Path) -> None:
    transcript = tmp_path / "uci.txt"
    transcript.write_text(
        "id name engine\nuciok\nreadyok\nbestmove g1f3\n", encoding="utf-8"
    )

    assert (
        validate_transcript(ValidationRequest(transcript, ("e2e4", "e7e5"))) == "g1f3"
    )


@pytest.mark.parametrize("bestmove", ["e2e4", "broken"])
def test_validate_transcript_rejects_illegal_or_invalid_bestmove(
    tmp_path: Path, bestmove: str
) -> None:
    transcript = tmp_path / "uci.txt"
    transcript.write_text(f"uciok\nreadyok\nbestmove {bestmove}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_transcript(ValidationRequest(transcript, ("e2e4", "e7e5")))
