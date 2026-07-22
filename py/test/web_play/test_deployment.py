from __future__ import annotations

from pathlib import Path

import pytest

from web_play.deployment import DeploymentConfiguration, download_model_artifacts

_RESOLVED_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _environment() -> dict[str, str]:
    return {
        "CHESS_MODEL_REPO_ID": "owner/chess-model",
        "CHESS_MODEL_REVISION": "main",
        "CHESS_MODEL_CHECKPOINT_FILENAME": "model.pt",
        "CHESS_MODEL_TORCHSCRIPT_FILENAME": "model.jit.pt",
        "CHESS_WEB_ALLOWED_ORIGINS": "https://chess.example, http://localhost:5173/",
    }


def test_deployment_configuration_accepts_explicit_latest_revision() -> None:
    configuration = DeploymentConfiguration.from_environment(_environment())
    assert configuration.hugging_face_revision == "main"
    assert configuration.allowed_origins == (
        "https://chess.example",
        "http://localhost:5173",
    )


def test_deployment_configuration_accepts_commit_revision() -> None:
    environment = _environment()
    environment["CHESS_MODEL_REVISION"] = _RESOLVED_REVISION
    configuration = DeploymentConfiguration.from_environment(environment)
    assert configuration.hugging_face_revision == _RESOLVED_REVISION


@pytest.mark.parametrize("revision", ["", "v1.0", "0123456"])
def test_deployment_configuration_rejects_ambiguous_revision(revision: str) -> None:
    environment = _environment()
    environment["CHESS_MODEL_REVISION"] = revision
    with pytest.raises(ValueError, match="CHESS_MODEL_REVISION"):
        DeploymentConfiguration.from_environment(environment)


def test_downloads_both_named_artifacts_and_returns_torchscript_path() -> None:
    downloaded_filenames: list[str] = []

    def downloader(
        *, repo_id: str, filename: str, revision: str, token: str | None
    ) -> str:
        assert repo_id == "owner/chess-model"
        assert revision == _RESOLVED_REVISION
        assert token == "secret"
        downloaded_filenames.append(filename)
        return str(Path("/cache") / filename)

    model_path = download_model_artifacts(
        DeploymentConfiguration.from_environment(_environment()),
        resolved_revision=_RESOLVED_REVISION,
        token="secret",
        downloader=downloader,
    )
    assert downloaded_filenames == ["model.pt", "model.jit.pt"]
    assert model_path == Path("/cache/model.jit.pt")


def test_download_rejects_an_unresolved_revision() -> None:
    def unused_downloader(
        *, repo_id: str, filename: str, revision: str, token: str | None
    ) -> str:
        raise AssertionError("The downloader must not be called.")

    with pytest.raises(ValueError, match="resolved Hugging Face revision"):
        download_model_artifacts(
            DeploymentConfiguration.from_environment(_environment()),
            resolved_revision="main",
            token=None,
            downloader=unused_downloader,
        )
