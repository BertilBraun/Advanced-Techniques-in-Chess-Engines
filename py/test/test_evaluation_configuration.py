from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from src.evaluation.configuration import EvaluationConfiguration
from src.evaluation.contracts import MatchEvaluationJob, OpeningLine, RandomOpponent
from src.experiment.configuration import load_experiment_configuration
from src.training.checkpoint import CheckpointReference


def test_checked_in_evaluation_definitions_are_canonical_for_each_game() -> None:
    chess = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    go = load_experiment_configuration(Path('configs/go-7x7-experiment-template.yaml'))

    assert chess.evaluation.engine.kind == 'stockfish'
    assert go.evaluation.engine.kind == 'katago'
    assert chess.evaluation.cadence_seconds == go.evaluation.cadence_seconds == 1200
    assert chess.evaluation.job_timeout_seconds == go.evaluation.job_timeout_seconds == 1200
    assert chess.training.topology.evaluation.device_cycle == (0,)
    assert chess.evaluation.maximum_concurrent_jobs == go.evaluation.maximum_concurrent_jobs == 10
    assert tuple(
        definition.boundary_offset
        for definition in chess.evaluation.definitions
        if definition.kind == 'previous_checkpoint'
    ) == (1, 2, 3)
    assert all(
        definition.kind != 'fixed_checkpoint'
        for definition in (*chess.evaluation.definitions, *go.evaluation.definitions)
    )
    assert tuple(
        definition.skill_level for definition in chess.evaluation.definitions if definition.kind == 'stockfish'
    ) == (0, 1, 2, 3)
    assert tuple(
        definition.maximum_visits for definition in go.evaluation.definitions if definition.kind == 'katago'
    ) == (16, 64, 128)
    assert tuple(
        (definition.boundary_offset, definition.boundary_parity)
        for definition in go.evaluation.definitions
        if definition.kind == 'previous_checkpoint'
    ) == (
        (1, 'every'),
        (2, 'every'),
        (3, 'every'),
        (4, 'even'),
        (5, 'odd'),
        (6, 'even'),
        (7, 'odd'),
        (8, 'even'),
        (9, 'odd'),
        (10, 'even'),
        (11, 'odd'),
    )
    assert all(definition.kind not in ('random', 'policy_random') for definition in go.evaluation.definitions)


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


def test_resolved_match_opponent_must_match_its_definition() -> None:
    experiment = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    stockfish_definition = experiment.evaluation.definitions[-1]
    checkpoint = CheckpointReference(
        generation=1,
        manifest_path=Path('checkpoint.json'),
        model_path=Path('model.pt'),
        optimizer_path=Path('optimizer.pt'),
        inference_model_path=Path('inference.pt'),
        inference_model_sha256='0' * 64,
    )

    with pytest.raises(ValidationError, match='do not match'):
        MatchEvaluationJob(
            kind='match',
            job_id='invalid',
            definition=stockfish_definition,
            boundary_seconds=1200,
            candidate=checkpoint,
            opponent=RandomOpponent(kind='random'),
            device_id=0,
            deadline_seconds=60,
            random_seed=0,
            result_path=Path('result.json'),
        )
