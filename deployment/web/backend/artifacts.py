from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_COMMIT_REVISION = re.compile(r'^[0-9a-f]{40}$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_LATEST_REVISION = 'main'


@dataclass(frozen=True)
class DeploymentConfiguration:
    hugging_face_repository_id: str
    hugging_face_revision: str
    inference_filename: str
    inference_sha256: str
    allowed_origins: tuple[str, ...]

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> DeploymentConfiguration:
        repository_id = _required(environment, 'CHESS_MODEL_REPO_ID')
        revision = _required(environment, 'CHESS_MODEL_REVISION')
        inference_filename = _required(environment, 'CHESS_MODEL_INFERENCE_FILENAME')
        inference_sha256 = _required(environment, 'CHESS_MODEL_SHA256')
        origins_text = _required(environment, 'CHESS_WEB_ALLOWED_ORIGINS')

        if '/' not in repository_id:
            raise ValueError('CHESS_MODEL_REPO_ID must be a namespace/repository id.')
        if revision != _LATEST_REVISION and _COMMIT_REVISION.fullmatch(revision) is None:
            raise ValueError("CHESS_MODEL_REVISION must be 'main' or a full 40-character commit hash.")
        if not inference_filename.endswith(('.jit.pt', '.onnx')):
            raise ValueError('CHESS_MODEL_INFERENCE_FILENAME must name a .jit.pt or .onnx artifact.')
        if _SHA256.fullmatch(inference_sha256) is None:
            raise ValueError('CHESS_MODEL_SHA256 must be a lowercase SHA-256 digest.')

        origins = tuple(origin.strip().rstrip('/') for origin in origins_text.split(',') if origin.strip())
        if not origins or any(origin == '*' for origin in origins):
            raise ValueError('CHESS_WEB_ALLOWED_ORIGINS must contain explicit browser origins.')
        return cls(repository_id, revision, inference_filename, inference_sha256, origins)


class ArtifactDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        token: str | None,
    ) -> str: ...


def download_model_artifact(
    configuration: DeploymentConfiguration,
    resolved_revision: str,
    token: str | None,
    downloader: ArtifactDownloader,
) -> Path:
    if _COMMIT_REVISION.fullmatch(resolved_revision) is None:
        raise ValueError('The resolved Hugging Face revision must be a commit hash.')
    downloaded_path = downloader(
        repo_id=configuration.hugging_face_repository_id,
        filename=configuration.inference_filename,
        revision=resolved_revision,
        token=token,
    )
    return Path(downloaded_path)


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if value is None or not value.strip():
        raise ValueError(f'Required deployment setting {name} is missing.')
    return value.strip()
