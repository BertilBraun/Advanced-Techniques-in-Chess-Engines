from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_COMMIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_LATEST_REVISION = "main"


@dataclass(frozen=True)
class DeploymentConfiguration:
    hugging_face_repository_id: str
    hugging_face_revision: str
    checkpoint_filename: str
    torchscript_filename: str
    allowed_origins: tuple[str, ...]

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str]
    ) -> DeploymentConfiguration:
        repository_id = _required(environment, "CHESS_MODEL_REPO_ID")
        revision = _required(environment, "CHESS_MODEL_REVISION")
        checkpoint_filename = _required(environment, "CHESS_MODEL_CHECKPOINT_FILENAME")
        torchscript_filename = _required(
            environment, "CHESS_MODEL_TORCHSCRIPT_FILENAME"
        )
        origins_text = _required(environment, "CHESS_WEB_ALLOWED_ORIGINS")

        if "/" not in repository_id:
            raise ValueError("CHESS_MODEL_REPO_ID must be a namespace/repository id.")
        if revision != _LATEST_REVISION and _COMMIT_REVISION.fullmatch(revision) is None:
            raise ValueError(
                "CHESS_MODEL_REVISION must be 'main' or a full 40-character commit hash."
            )
        if not checkpoint_filename.endswith(".pt") or checkpoint_filename.endswith(
            ".jit.pt"
        ):
            raise ValueError(
                "CHESS_MODEL_CHECKPOINT_FILENAME must name the training .pt artifact."
            )
        if not torchscript_filename.endswith(".jit.pt"):
            raise ValueError(
                "CHESS_MODEL_TORCHSCRIPT_FILENAME must name the .jit.pt artifact."
            )

        origins = tuple(
            origin.strip().rstrip("/")
            for origin in origins_text.split(",")
            if origin.strip()
        )
        if not origins or any(origin == "*" for origin in origins):
            raise ValueError(
                "CHESS_WEB_ALLOWED_ORIGINS must contain explicit browser origins."
            )
        return cls(
            repository_id, revision, checkpoint_filename, torchscript_filename, origins
        )


class ArtifactDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        token: str | None,
    ) -> str: ...


def download_model_artifacts(
    configuration: DeploymentConfiguration,
    resolved_revision: str,
    token: str | None,
    downloader: ArtifactDownloader,
) -> Path:
    if _COMMIT_REVISION.fullmatch(resolved_revision) is None:
        raise ValueError("The resolved Hugging Face revision must be a commit hash.")
    torchscript_path: Path | None = None
    for filename in (
        configuration.checkpoint_filename,
        configuration.torchscript_filename,
    ):
        downloaded_path = downloader(
            repo_id=configuration.hugging_face_repository_id,
            filename=filename,
            revision=resolved_revision,
            token=token,
        )
        if filename == configuration.torchscript_filename:
            torchscript_path = Path(downloaded_path)
    assert torchscript_path is not None
    return torchscript_path


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Required deployment setting {name} is missing.")
    return value.strip()
