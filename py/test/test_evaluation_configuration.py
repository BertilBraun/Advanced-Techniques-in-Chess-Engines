from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from src.evaluation.configuration import (
    EvaluationConfiguration,
    StockfishEngineConfiguration,
    StockfishFixedNodesEvaluationDefinition,
)
from src.evaluation.contracts import (
    EvaluationJob,
    MatchEvaluationJob,
    OpeningLine,
    RandomOpponent,
    StockfishFixedNodesOpponent,
)
from src.evaluation.process import PROJECT_ROOT, stockfish_fixed_nodes_executable_path
from src.evaluation.scheduling import ScheduledEvaluationSuite, jobs_for_suite
from src.experiment.configuration import load_experiment_configuration
from test_helpers.checkpoints import checkpoint_reference
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY


def test_evaluation_definition_ids_must_be_unique() -> None:
    experiment = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    payload = experiment.evaluation.model_dump(mode='json')
    payload['definitions'][1]['definition_id'] = payload['definitions'][0]['definition_id']

    with pytest.raises(ValidationError, match='must be unique'):
        EvaluationConfiguration.model_validate(payload)


def test_opening_suite_must_cover_largest_match_definition() -> None:
    experiment = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'go-7x7-experiment.yaml')
    payload = experiment.evaluation.model_dump(mode='json')
    payload['openings']['opening_count'] = 1
    payload['definitions'][1]['opening_pair_count'] = 2

    with pytest.raises(ValidationError, match='cover every requested opening pair'):
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
    experiment = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    stockfish_definition = experiment.evaluation.definitions[-1]
    checkpoint = checkpoint_reference()

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


def test_stockfish_fixed_nodes_definition_round_trips_and_schedules_distinct_opponent(tmp_path: Path) -> None:
    experiment = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    skill_definition = next(
        definition for definition in experiment.evaluation.definitions if definition.kind == 'stockfish'
    )
    definition_payload = skill_definition.model_dump(mode='json')
    definition_payload['kind'] = 'stockfish_fixed_nodes'
    definition_payload['definition_id'] = 'stockfish-fixed-nodes'
    definition_payload['nodes'] = 300
    del definition_payload['skill_level']
    evaluation_payload = experiment.evaluation.model_dump(mode='json')
    evaluation_payload['definitions'].append(definition_payload)
    evaluation = EvaluationConfiguration.model_validate(evaluation_payload)
    round_tripped = EvaluationConfiguration.model_validate_json(evaluation.model_dump_json())
    fixed_definition = round_tripped.definitions[-1]

    assert isinstance(fixed_definition, StockfishFixedNodesEvaluationDefinition)
    assert fixed_definition.nodes == 300
    assert 'skill_level' not in fixed_definition.model_dump()

    configured_experiment = experiment.model_copy(update={'evaluation': round_tripped})
    checkpoint = checkpoint_reference(tmp_path)
    jobs, _ = jobs_for_suite(
        configured_experiment,
        tmp_path,
        tmp_path / 'results',
        ScheduledEvaluationSuite(boundary_seconds=1200, checkpoint=checkpoint),
        (),
        0,
    )
    job = next(job for job in jobs if job.definition.kind == 'stockfish_fixed_nodes')

    assert isinstance(job, MatchEvaluationJob)
    assert isinstance(job.opponent, StockfishFixedNodesOpponent)
    assert job.opponent.nodes == 300
    serialized_job = TypeAdapter(EvaluationJob).validate_json(job.model_dump_json()).model_dump(mode='json')
    assert serialized_job['opponent'] == {
        'kind': 'stockfish_fixed_nodes',
        'nodes': 300,
        'engine_executable_path': None,
    }


def test_stockfish_fixed_nodes_engine_override_reaches_opponent_and_executable(tmp_path: Path) -> None:
    experiment = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    skill_definition = next(
        definition for definition in experiment.evaluation.definitions if definition.kind == 'stockfish'
    )
    definition_payload = skill_definition.model_dump(mode='json')
    definition_payload['kind'] = 'stockfish_fixed_nodes'
    definition_payload['definition_id'] = 'stockfish-fixed-nodes'
    definition_payload['nodes'] = 300
    definition_payload['engine_executable_path'] = 'engines/stockfish-13'
    del definition_payload['skill_level']
    evaluation_payload = experiment.evaluation.model_dump(mode='json')
    evaluation_payload['definitions'].append(definition_payload)
    evaluation = EvaluationConfiguration.model_validate(evaluation_payload)
    configured_experiment = experiment.model_copy(update={'evaluation': evaluation})
    assert isinstance(evaluation.engine, StockfishEngineConfiguration)

    jobs, _ = jobs_for_suite(
        configured_experiment,
        tmp_path,
        tmp_path / 'results',
        ScheduledEvaluationSuite(boundary_seconds=1200, checkpoint=checkpoint_reference(tmp_path)),
        (),
        0,
    )
    job = next(job for job in jobs if job.definition.kind == 'stockfish_fixed_nodes')

    assert isinstance(job, MatchEvaluationJob)
    assert isinstance(job.opponent, StockfishFixedNodesOpponent)
    assert job.opponent.engine_executable_path == 'engines/stockfish-13'
    assert stockfish_fixed_nodes_executable_path(evaluation.engine, job.opponent) == PROJECT_ROOT / Path(
        'engines/stockfish-13'
    )
    default_opponent = StockfishFixedNodesOpponent(kind='stockfish_fixed_nodes', nodes=300)
    assert stockfish_fixed_nodes_executable_path(evaluation.engine, default_opponent) == PROJECT_ROOT / Path(
        evaluation.engine.executable_path
    )


def test_stockfish_fixed_node_rungs_require_unique_node_counts() -> None:
    experiment = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    skill_definition = next(
        definition for definition in experiment.evaluation.definitions if definition.kind == 'stockfish'
    )
    evaluation_payload = experiment.evaluation.model_dump(mode='json')
    for definition_id in ('rung-a', 'rung-b'):
        definition_payload = skill_definition.model_dump(mode='json')
        definition_payload['kind'] = 'stockfish_fixed_nodes'
        definition_payload['definition_id'] = definition_id
        definition_payload['nodes'] = 100
        del definition_payload['skill_level']
        evaluation_payload['definitions'].append(definition_payload)

    with pytest.raises(ValidationError, match='unique node counts'):
        EvaluationConfiguration.model_validate(evaluation_payload)
