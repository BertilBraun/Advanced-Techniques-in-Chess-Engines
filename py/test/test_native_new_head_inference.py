from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

pytest.importorskip('AlphaZeroCpp')
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.games.chess.interactive.analysis import CountedMctsAnalysis, PolicyAnalysis
from src.games.chess.interactive.configuration import InferenceTarget, InteractiveEngineConfiguration
from src.games.chess.interactive.engine import InteractiveEngine
from src.training.checkpoint.persistence import create_optimizer, save_model_and_optimizer
from src.training.network import (
    AttentionNetworkParams,
    Chess76PlaneDirectPolicyHeadConfiguration,
    DisabledResidualContext,
    Network,
    NetworkConfiguration,
    NetworkParams,
)
from src.training.targets import NextPolicyHeadLayout

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    'parameters',
    (
        NetworkParams(
            num_layers=1,
            hidden_size=64,
            residual_context=DisabledResidualContext(),
            policy_head=Chess76PlaneDirectPolicyHeadConfiguration(),
            num_value_channels=2,
            value_fc_size=16,
        ),
        AttentionNetworkParams(
            num_layers=1,
            embedding_size=64,
            num_heads=2,
            feedforward_size=128,
            policy_head=Chess76PlaneDirectPolicyHeadConfiguration(),
            num_value_channels=2,
            value_fc_size=16,
        ),
    ),
)
def test_native_inference_pipeline_searches_with_an_exported_new_head_model(
    tmp_path: Path,
    parameters: NetworkConfiguration,
) -> None:
    torch.manual_seed(19)
    auxiliary_heads = (NextPolicyHeadLayout(kind='next_policy', action_size=4864, ply_offset=1),)
    model = Network(parameters, torch.device('cpu'), CHESS_NETWORK_DIMENSIONS, auxiliary_heads)
    save_model_and_optimizer(model, create_optimizer(model, 'adamw'), 0, tmp_path)

    engine = InteractiveEngine(
        InteractiveEngineConfiguration(
            model_path=str(tmp_path / 'model_0.jit.pt'),
            device_id=0,
            parallel_searches=2,
            exploration_constant=1.0,
            maximum_batch_size=2,
            outstanding_batches_per_worker=2,
            inference_target=InferenceTarget.CPU,
        )
    )
    game = engine.new_game(chess.STARTING_FEN, ())

    policy_result = game.analyze(PolicyAnalysis())
    search_result = game.analyze(CountedMctsAnalysis(searches=16))

    legal_moves = {move.uci() for move in chess.Board().legal_moves}
    assert {candidate.move_uci for candidate in policy_result.candidates} == legal_moves
    assert policy_result.outcome is not None
    assert sum(candidate.policy_prior for candidate in policy_result.candidates) == pytest.approx(1.0, abs=1e-4)
    assert search_result.searches >= 16
    assert search_result.chosen_move_uci in legal_moves
