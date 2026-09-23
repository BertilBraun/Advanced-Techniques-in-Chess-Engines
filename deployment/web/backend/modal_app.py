from __future__ import annotations

import os
import time
from pathlib import Path

import chess
import modal
from fastapi import FastAPI

_REMOTE_ROOT = '/opt/chess'
_TENSORRT_CACHE_ROOT = '/cache/tensorrt'
_GPU_SINGLE_POSITION_WARMUPS = 2
_GPU_WARMUP_SEARCHES = 256
_INFERENCE_BATCH_SIZE = 64
_INPUT_CHANNELS = 52
_BOARD_ROWS = 8
_BOARD_COLUMNS = 8
_TENSORRT_BUILDER_OPTIMIZATION_LEVEL = 3
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path(_REMOTE_ROOT)

image = (
    modal.Image.from_registry(
        'nvcr.io/nvidia/tensorrt:25.11-py3',
        add_python='3.12',
    )
    .entrypoint([])
    .apt_install('build-essential', 'cmake', 'git')
    .pip_install(
        'torch==2.12.1',
        index_url='https://download.pytorch.org/whl/cu126',
    )
    .pip_install_from_pyproject(
        str(_REPOSITORY_ROOT / 'pyproject.toml'),
        optional_dependencies=['web', 'tensorrt'],
    )
    .pip_install('ruff==0.16.4')
    .add_local_dir(
        _REPOSITORY_ROOT / 'cpp',
        f'{_REMOTE_ROOT}/cpp',
        copy=True,
        ignore=['build*/**', 'libtorch*/**'],
    )
    .run_commands(
        f'cmake -S {_REMOTE_ROOT}/cpp -B {_REMOTE_ROOT}/cpp/build '
        '-DCMAKE_BUILD_TYPE=Release -DENABLE_NATIVE_ARCHITECTURE=OFF -DENABLE_TENSORRT=ON',
        f'cmake --build {_REMOTE_ROOT}/cpp/build --parallel 2',
        f'ctest --test-dir {_REMOTE_ROOT}/cpp/build --output-on-failure',
    )
    .add_local_dir(
        _REPOSITORY_ROOT / 'py',
        f'{_REMOTE_ROOT}/py',
        copy=True,
        ignore=['test/**', '.pytest_cache/**', '**/__pycache__/**', '*.pt', '*.pyd', '*.so'],
    )
    .add_local_dir(
        _REPOSITORY_ROOT / 'deployment',
        f'{_REMOTE_ROOT}/deployment',
        copy=True,
        ignore=['web/frontend/**', '**/__pycache__/**'],
    )
    .env({'PYTHONPATH': f'{_REMOTE_ROOT}:{_REMOTE_ROOT}/py'})
)

app = modal.App('chess-model-web-play')
tensor_rt_cache = modal.Volume.from_name('chess-web-play-tensorrt-cache', create_if_missing=True)


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name('chess-web-play')],
    min_containers=0,
    max_containers=1,
    scaledown_window=120,
    gpu='A10',
    cpu=2.0,
    memory=2048,
    timeout=90,
    startup_timeout=900,
    volumes={_TENSORRT_CACHE_ROOT: tensor_rt_cache},
)
@modal.concurrent(max_inputs=1)
class ChessWebPlay:
    @modal.enter()
    def load_engine(self) -> None:
        startup_started = time.perf_counter()
        import torch
        from src.games.chess.interactive.analysis import CountedMctsAnalysis, PolicyAnalysis
        from src.games.chess.interactive.configuration import (
            InferenceTarget,
            InteractiveEngineConfiguration,
            InteractiveInferenceBackend,
        )
        from src.games.chess.interactive.engine import InteractiveEngine

        from deployment.web.backend.api import create_app
        from deployment.web.backend.artifacts import DeploymentConfiguration
        from deployment.web.backend.service import GameService
        from deployment.web.backend.tensorrt_cache import (
            build_cached_tensorrt_engine,
            create_tensorrt_cache_identity,
            find_cached_tensorrt_engine,
        )
        from deployment.web.backend.tensorrt_identity import modal_runtime_identity

        configuration = DeploymentConfiguration.from_environment(os.environ)
        if not torch.cuda.is_available():
            raise RuntimeError('The GPU deployment cannot access CUDA.')
        cache_identity = create_tensorrt_cache_identity(
            source_sha256=configuration.inference_sha256,
            runtime=modal_runtime_identity(),
            batch_size=_INFERENCE_BATCH_SIZE,
            channels=_INPUT_CHANNELS,
            rows=_BOARD_ROWS,
            columns=_BOARD_COLUMNS,
            builder_optimization_level=_TENSORRT_BUILDER_OPTIMIZATION_LEVEL,
        )
        cached_engine = find_cached_tensorrt_engine(Path(_TENSORRT_CACHE_ROOT), cache_identity)
        if cached_engine is None:
            from huggingface_hub import HfApi, hf_hub_download
            from src.util.hashing import file_sha256

            from deployment.web.backend.artifacts import download_model_artifact
            from deployment.web.backend.tensorrt_runtime import build_and_verify_tensorrt_engine

            hugging_face_token = os.environ.get('HF_TOKEN')
            model_information = HfApi().model_info(
                repo_id=configuration.hugging_face_repository_id,
                revision=configuration.hugging_face_revision,
                token=hugging_face_token,
            )
            resolved_revision = model_information.sha
            if resolved_revision is None:
                raise ValueError('Hugging Face returned no resolved model revision.')
            model_path = download_model_artifact(
                configuration=configuration,
                resolved_revision=resolved_revision,
                token=hugging_face_token,
                downloader=hf_hub_download,
            )
            if file_sha256(model_path) != configuration.inference_sha256:
                raise ValueError('The downloaded inference artifact does not match CHESS_MODEL_SHA256.')
            cached_engine = build_cached_tensorrt_engine(
                source_path=model_path,
                cache_root=Path(_TENSORRT_CACHE_ROOT),
                identity=cache_identity,
                build_engine=build_and_verify_tensorrt_engine,
            )
            tensor_rt_cache.commit()
        cache_status = 'built and cached' if cached_engine.built else 'loaded from cache'
        print(
            f'Loading {cache_status} TensorRT engine on {torch.cuda.get_device_name(0)} '
            f'after {time.perf_counter() - startup_started:.1f} seconds.'
        )
        engine = InteractiveEngine(
            InteractiveEngineConfiguration(
                model_path=str(cached_engine.path),
                parallel_searches=16,
                maximum_batch_size=_INFERENCE_BATCH_SIZE,
                inference_target=InferenceTarget.CUDA,
                inference_backend=InteractiveInferenceBackend.TENSORRT,
            )
        )
        for _ in range(_GPU_SINGLE_POSITION_WARMUPS):
            engine.new_game(chess.STARTING_FEN, ()).analyze(PolicyAnalysis())
        warmup_result = engine.new_game(chess.STARTING_FEN, ()).analyze(
            CountedMctsAnalysis(searches=_GPU_WARMUP_SEARCHES)
        )
        print(
            f'Warmed TensorRT inference with {warmup_result.searches} searches '
            f'in {warmup_result.elapsed_milliseconds} ms; startup completed in '
            f'{time.perf_counter() - startup_started:.1f} seconds.'
        )
        self._web_application = create_app(GameService(engine), configuration.allowed_origins)

    @modal.asgi_app()
    def web(self) -> FastAPI:
        return self._web_application
