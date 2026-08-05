from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from src.interactive.analysis import AnalysisRequest, CountedMctsAnalysis, PolicyAnalysis, TimedMctsAnalysis
from src.interactive.configuration import InteractiveEngineConfiguration


def test_configuration_resolves_batch_size_from_parallel_searches() -> None:
    configuration = InteractiveEngineConfiguration(model_path='model.jit.pt', parallel_searches=32)

    assert configuration.resolved_batch_size == 32


@pytest.mark.parametrize(
    'configuration',
    (
        {'model_path': ''},
        {'model_path': 'model.jit.pt', 'parallel_searches': 0},
        {'model_path': 'model.jit.pt', 'outstanding_batches_per_worker': 3},
    ),
)
def test_configuration_rejects_invalid_values(configuration: dict[str, str | int]) -> None:
    with pytest.raises(ValueError):
        InteractiveEngineConfiguration(**configuration)


@pytest.mark.parametrize(
    ('payload', 'expected'),
    (
        ({'type': 'policy'}, PolicyAnalysis()),
        ({'type': 'timed_mcts', 'seconds': 5}, TimedMctsAnalysis(seconds=5)),
        ({'type': 'counted_mcts', 'searches': 128}, CountedMctsAnalysis(searches=128)),
    ),
)
def test_analysis_request_variants_validate(payload: dict[str, str | int], expected: AnalysisRequest) -> None:
    assert TypeAdapter(AnalysisRequest).validate_python(payload) == expected


def test_analysis_request_rejects_irrelevant_fields() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AnalysisRequest).validate_python({'type': 'policy', 'seconds': 1})
