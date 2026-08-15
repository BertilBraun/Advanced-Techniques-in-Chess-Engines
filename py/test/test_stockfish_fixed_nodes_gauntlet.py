from pathlib import Path

from src.evaluation.configuration import StockfishFixedNodesEvaluationDefinition
from src.experiment.configuration import load_chess_experiment_configuration
from tools.run_stockfish_fixed_nodes_gauntlet import _fixed_nodes_definition, _stockfish_configuration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = PROJECT_ROOT / 'py/configs/production/vast-chess-8gpu-1d-r3.yaml'


def test_gauntlet_reuses_production_search_and_overrides_match_identity() -> None:
    configuration = load_chess_experiment_configuration(EXPERIMENT_PATH)

    definition = _fixed_nodes_definition(configuration, opening_pairs=25, match_nodes=2_000)

    assert isinstance(definition, StockfishFixedNodesEvaluationDefinition)
    assert definition.definition_id == 'stockfish-13-fixed-nodes-2000'
    assert definition.opening_pair_count == 25
    assert definition.search.searches_per_move == 64
    assert definition.search.parallel_searches == 1
    assert definition.search.exploration_constant == 1.0
    assert definition.search.inference.inference_workers == 1
    assert definition.search.inference.inference_batch_size == 64
    assert definition.search.inference.outstanding_batches_per_worker == 1


def test_gauntlet_preserves_stockfish_resources_and_overrides_nodes(tmp_path: Path) -> None:
    configuration = load_chess_experiment_configuration(EXPERIMENT_PATH)
    executable = tmp_path / 'stockfish-13'

    engine = _stockfish_configuration(configuration, executable, match_nodes=5_000)

    assert engine.executable_path == str(executable.resolve())
    assert engine.match_nodes == 5_000
    assert engine.threads == 1
    assert engine.hash_mib == 1_024
