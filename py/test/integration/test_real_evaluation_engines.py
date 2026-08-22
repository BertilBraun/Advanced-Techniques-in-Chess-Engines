from pathlib import Path
import os

import pytest

from src.evaluation.configuration import KataGoEngineConfiguration, StockfishEngineConfiguration
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.chess.stockfish import StockfishClient
from src.games.go.contract import GoStateContract
from src.games.go.katago import KataGoClient


@pytest.mark.integration
def test_real_stockfish_returns_an_initial_policy() -> None:
    configured_path = os.environ.get('EVALUATION_STOCKFISH_EXECUTABLE')
    if configured_path is None:
        pytest.skip('EVALUATION_STOCKFISH_EXECUTABLE is not configured.')
    executable_path = Path(configured_path)
    configuration = StockfishEngineConfiguration(
        kind='stockfish',
        executable_path=str(executable_path),
        label_nodes=100,
        match_nodes=100,
        threads=1,
        hash_mib=16,
        multi_pv=2,
        policy_softmax_temperature=0.15,
    )

    with StockfishClient(configuration, CHESS_STATE_CONTRACT, executable_path) as client:
        policy = client.policy(CHESS_STATE_CONTRACT.initial_position(), ())

    assert policy.entries
    assert sum(entry.probability for entry in policy.entries) == pytest.approx(1.0)


@pytest.mark.integration
def test_real_katago_returns_an_initial_policy() -> None:
    executable_value = os.environ.get('EVALUATION_KATAGO_EXECUTABLE')
    model_value = os.environ.get('EVALUATION_KATAGO_MODEL')
    configuration_value = os.environ.get('EVALUATION_KATAGO_CONFIGURATION')
    if executable_value is None or model_value is None or configuration_value is None:
        pytest.skip('KataGo evaluation artifact environment variables are not configured.')
    executable = Path(executable_value)
    model = Path(model_value)
    analysis_configuration = Path(configuration_value)
    state = GoStateContract(7, komi_half_points=15, maximum_moves=196)
    configuration = KataGoEngineConfiguration(
        kind='katago',
        executable_path=str(executable),
        model_path=str(model),
        analysis_configuration_path=str(analysis_configuration),
        label_max_visits=16,
    )

    with KataGoClient(configuration, state, executable, model, analysis_configuration) as client:
        policy = client.policy(state.initial_position(), ())

    assert policy.entries
    assert sum(entry.probability for entry in policy.entries) == pytest.approx(1.0)
