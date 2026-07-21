from __future__ import annotations

import os
from pathlib import Path

import modal
from fastapi import FastAPI

_REMOTE_ROOT = "/opt/chess"
_REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2] if modal.is_local() else Path(_REMOTE_ROOT)
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "cmake", "git")
    .pip_install_from_requirements(
        str(_REPOSITORY_ROOT / "py" / "requirements-web-modal.txt")
    )
    .pip_install(
        "torch==2.7.1+cpu",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_dir(
        _REPOSITORY_ROOT / "cpp",
        f"{_REMOTE_ROOT}/cpp",
        copy=True,
        ignore=["build/**", "libtorch*/**"],
    )
    .add_local_dir(
        _REPOSITORY_ROOT / "py",
        f"{_REMOTE_ROOT}/py",
        copy=True,
        ignore=["test/**", "**/__pycache__/**", "*.pt", "*.pyd", "*.so"],
    )
    .run_commands(
        f"cmake -S {_REMOTE_ROOT}/cpp -B {_REMOTE_ROOT}/cpp/build -DCMAKE_BUILD_TYPE=Release",
        f"cmake --build {_REMOTE_ROOT}/cpp/build --parallel 2",
    )
    .env({"PYTHONPATH": f"{_REMOTE_ROOT}/py"})
)

app = modal.App("chess-model-web-play")


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("chess-web-play")],
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=90,
    startup_timeout=900,
)
@modal.concurrent(max_inputs=1)
class ChessWebPlay:
    @modal.enter()
    def load_engine(self) -> None:
        from huggingface_hub import hf_hub_download

        from web_play.api import create_app
        from web_play.deployment import (
            DeploymentConfiguration,
            download_model_artifacts,
        )
        from web_play.native_adapter import (
            NativeEngineConfiguration,
            NativeInteractiveEngine,
        )
        from web_play.service import GameService

        configuration = DeploymentConfiguration.from_environment(os.environ)
        model_path = download_model_artifacts(
            configuration=configuration,
            token=os.environ.get("HF_TOKEN"),
            downloader=hf_hub_download,
        )
        engine = NativeInteractiveEngine(
            NativeEngineConfiguration.for_model(str(model_path))
        )
        self._web_application = create_app(
            GameService(engine), configuration.allowed_origins
        )

    @modal.asgi_app()
    def web(self) -> FastAPI:
        return self._web_application
