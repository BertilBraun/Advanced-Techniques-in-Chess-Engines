from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from src.evaluation.configuration import EvaluationConfiguration
from src.evaluation.contracts import OpeningLine
from src.experiment.configuration import load_experiment_configuration


def test_checked_in_evaluation_definitions_are_canonical_for_each_game() -> None:
    chess = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    go = load_experiment_configuration(Path('configs/go-7x7-experiment-template.yaml'))

    assert chess.evaluation.engine.kind == 'stockfish'
    assert go.evaluation.engine.kind == 'katago'
    assert chess.evaluation.cadence_seconds == go.evaluation.cadence_seconds == 1200
    assert chess.training.topology.evaluation.device_cycle == (0,)
    assert tuple(definition.definition_id for definition in chess.evaluation.definitions) == (
        'fixed-dataset',
        'random',
        'previous-checkpoint',
        'stockfish-level-0',
    )


def test_evaluation_definition_ids_must_be_unique() -> None:
    experiment = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    payload = experiment.evaluation.model_dump(mode='json')
    payload['definitions'][1]['definition_id'] = payload['definitions'][0]['definition_id']

    with pytest.raises(ValidationError, match='must be unique'):
        EvaluationConfiguration.model_validate(payload)


def test_opening_line_requires_exactly_four_plies() -> None:
    payload = {
        'opening_id': 'opening-0',
        'action_ids': [1, 2, 3],
        'path_probability': 0.5,
        'final_position_digest': '0' * 64,
        'human_readable': 'example',
    }

    with pytest.raises(ValidationError, match='at least 4'):
        TypeAdapter(OpeningLine).validate_python(payload)
