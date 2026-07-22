from pathlib import Path

import pytest

from tools.fetch_hf_model import (
    DownloadedModel,
    DownloadRequest,
    validate_request,
    write_provenance,
)


@pytest.mark.parametrize("revision", [None, "main", "a" * 40])
def test_validate_request_accepts_latest_or_selected_revision(
    revision: str | None,
) -> None:
    validate_request(
        DownloadRequest(
            "owner/repository",
            revision,
            "models/best.jit.pt",
            Path("models"),
            None,
        )
    )


@pytest.mark.parametrize(
    "filename",
    [
        "best.pt",
        "../best.jit.pt",
    ],
)
def test_validate_request_rejects_non_torchscript_or_unsafe_model(
    filename: str,
) -> None:
    with pytest.raises(ValueError):
        validate_request(
            DownloadRequest("owner/repository", None, filename, Path("models"), None)
        )


def test_write_provenance_records_resolved_revision(tmp_path: Path) -> None:
    provenance_path = tmp_path / "model-source.txt"
    request = DownloadRequest(
        "BertilBraun/alphazero-chess",
        None,
        "latest.jit.pt",
        tmp_path,
        provenance_path,
    )

    write_provenance(
        request,
        DownloadedModel(tmp_path / "latest.jit.pt", "a" * 40),
    )

    assert "requested_revision=latest" in provenance_path.read_text(encoding="utf-8")
    assert f"resolved_revision={'a' * 40}" in provenance_path.read_text(
        encoding="utf-8"
    )
