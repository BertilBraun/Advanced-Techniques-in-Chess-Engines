from __future__ import annotations

from pathlib import Path

from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.training import ChessImplementation
from src.self_play.configuration import TorchScriptInferenceBackend
from src.training.checkpoint.contracts import CheckpointReference

EXPERIMENT_PATH = Path(__file__).resolve().parents[2] / 'py/configs/production/vast-chess-8gpu-optimal.yaml'


def test_torchscript_inference_serves_a_checkpoint_outside_the_configured_models(tmp_path: Path) -> None:
    game = ChessImplementation(load_chess_experiment_configuration(EXPERIMENT_PATH))
    inference = game.self_play_configuration.inference.model_copy(update={'backend': TorchScriptInferenceBackend()})
    # No manifest exists, so resolving must not need the checkpoint's architecture at all.
    checkpoint = CheckpointReference(
        generation=7,
        manifest_path=tmp_path / 'checkpoint_7.json',
        model_path=tmp_path / 'model_7.pt',
        optimizer_path=tmp_path / 'optimizer_7.pt',
        inference_model_path=tmp_path / 'model_7.jit.pt',
        inference_model_sha256='0' * 64,
    )

    assert game.resolved_inference_model_path(checkpoint, inference) == checkpoint.inference_model_path
