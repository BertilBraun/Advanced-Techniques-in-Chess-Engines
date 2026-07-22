from __future__ import annotations

import os
from pathlib import Path

import chess
import modal
from fastapi import FastAPI

_REMOTE_ROOT = "/opt/chess"
_GPU_SINGLE_POSITION_WARMUPS = 2
_GPU_WARMUP_SEARCHES = 4096
_REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2] if modal.is_local() else Path(_REMOTE_ROOT)
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install("build-essential", "cmake", "git")
    .pip_install_from_requirements(
        str(_REPOSITORY_ROOT / "py" / "requirements-web-modal.txt")
    )
    .pip_install(
        "torch==2.7.1",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .add_local_dir(
        _REPOSITORY_ROOT / "cpp",
        f"{_REMOTE_ROOT}/cpp",
        copy=True,
        ignore=["build/**", "libtorch*/**"],
    )
    .run_commands(
        f"cmake -S {_REMOTE_ROOT}/cpp -B {_REMOTE_ROOT}/cpp/build "
        "-DCMAKE_BUILD_TYPE=Release -DENABLE_NATIVE_ARCHITECTURE=OFF",
        f"cmake --build {_REMOTE_ROOT}/cpp/build --parallel 2",
        f"ctest --test-dir {_REMOTE_ROOT}/cpp/build --output-on-failure",
    )
    .add_local_dir(
        _REPOSITORY_ROOT / "py",
        f"{_REMOTE_ROOT}/py",
        copy=True,
        ignore=["test/**", "**/__pycache__/**", "*.pt", "*.pyd", "*.so"],
    )
    .env({"PYTHONPATH": f"{_REMOTE_ROOT}/py"})
)

app = modal.App("chess-model-web-play")


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("chess-web-play")],
    min_containers=0,
    max_containers=1,
    scaledown_window=120,
    gpu="A10",
    cpu=4.0,
    memory=4096,
    timeout=90,
    startup_timeout=900,
)
@modal.concurrent(max_inputs=1)
class ChessWebPlay:
    @modal.enter()
    def load_engine(self) -> None:
        import torch
        from huggingface_hub import HfApi, hf_hub_download

        from src.eval.InteractiveEngine import InferenceTarget
        from web_play.api import create_app
        from web_play.contracts import AnalysisMode, CountedAnalysis
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
        hugging_face_token = os.environ.get("HF_TOKEN")
        model_information = HfApi().model_info(
            repo_id=configuration.hugging_face_repository_id,
            revision=configuration.hugging_face_revision,
            token=hugging_face_token,
        )
        resolved_revision = model_information.sha
        if resolved_revision is None:
            raise ValueError("Hugging Face returned no resolved model revision.")
        model_path = download_model_artifacts(
            configuration=configuration,
            resolved_revision=resolved_revision,
            token=hugging_face_token,
            downloader=hf_hub_download,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("The GPU deployment cannot access CUDA.")
        print(f"Loading interactive engine on {torch.cuda.get_device_name(0)}.")
        engine = NativeInteractiveEngine(
            NativeEngineConfiguration.for_model(
                str(model_path), inference_target=InferenceTarget.CUDA
            )
        )
        for _ in range(_GPU_SINGLE_POSITION_WARMUPS):
            engine.new_game(chess.STARTING_FEN, ()).analyze(
                CountedAnalysis(mode=AnalysisMode.MCTS, searches=1)
            )
        warmup_result = engine.new_game(chess.STARTING_FEN, ()).analyze(
            CountedAnalysis(mode=AnalysisMode.MCTS, searches=_GPU_WARMUP_SEARCHES)
        )
        print(
            f"Warmed CUDA inference with {warmup_result.metrics.searches} searches "
            f"in {warmup_result.metrics.elapsed_milliseconds} ms."
        )
        self._web_application = create_app(
            GameService(engine), configuration.allowed_origins
        )

    @modal.asgi_app()
    def web(self) -> FastAPI:
        return self._web_application
