from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_REPOSITORY = "BertilBraun/alphazero-chess"
DEFAULT_FILENAME = "latest.jit.pt"


@dataclass(frozen=True)
class DownloadRequest:
    repository: str
    revision: str | None
    filename: str
    output_directory: Path
    provenance_path: Path | None


@dataclass(frozen=True)
class DownloadedModel:
    path: Path
    resolved_revision: str


def parse_arguments() -> DownloadRequest:
    parser = argparse.ArgumentParser(
        description="Fetch a TorchScript model from Hugging Face."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision")
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    arguments = parser.parse_args()
    request = DownloadRequest(
        repository=arguments.repository,
        revision=arguments.revision,
        filename=arguments.filename,
        output_directory=arguments.output_directory,
        provenance_path=arguments.provenance,
    )
    validate_request(request)
    return request


def validate_request(request: DownloadRequest) -> None:
    if request.revision == "":
        raise ValueError(
            "revision must be omitted or contain a branch, tag, or commit."
        )
    if not request.filename.endswith(".jit.pt"):
        raise ValueError("filename must identify a TorchScript .jit.pt model.")
    if Path(request.filename).is_absolute() or ".." in Path(request.filename).parts:
        raise ValueError("filename must be a safe repository-relative path.")


def download_model(request: DownloadRequest) -> DownloadedModel:
    token = os.environ.get("HF_TOKEN")
    model_information = HfApi(token=token).model_info(
        repo_id=request.repository,
        revision=request.revision,
    )
    resolved_revision = model_information.sha
    if not COMMIT_PATTERN.fullmatch(resolved_revision):
        raise ValueError("Hugging Face did not resolve the model to a commit SHA.")
    downloaded_path = hf_hub_download(
        repo_id=request.repository,
        revision=resolved_revision,
        filename=request.filename,
        local_dir=request.output_directory,
        token=token,
    )
    result = DownloadedModel(Path(downloaded_path), resolved_revision)
    if request.provenance_path is not None:
        write_provenance(request, result)
    return result


def write_provenance(request: DownloadRequest, model: DownloadedModel) -> None:
    assert request.provenance_path is not None
    requested_revision = request.revision or "latest"
    request.provenance_path.parent.mkdir(parents=True, exist_ok=True)
    request.provenance_path.write_text(
        "\n".join(
            (
                f"repository={request.repository}",
                f"requested_revision={requested_revision}",
                f"resolved_revision={model.resolved_revision}",
                f"filename={request.filename}",
                f"path={model.path}",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> None:
    request = parse_arguments()
    downloaded_model = download_model(request)
    print(downloaded_model.path)
    print(f"Resolved Hugging Face revision: {downloaded_model.resolved_revision}")


if __name__ == "__main__":
    main()
