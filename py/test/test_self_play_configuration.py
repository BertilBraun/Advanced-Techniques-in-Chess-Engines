from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.self_play.configuration import BatchedInferenceParams, SdpaBackend, SelfPlaySearchParams


def _search_params_payload() -> dict[str, object]:
    return {
        'full_search_budget': {
            'kind': 'fixed',
            'visits': {
                'kind': 'staged',
                'stages': [
                    {'start_generation': 0, 'value': 200},
                    {'start_generation': 30, 'value': 600},
                ],
            },
        },
        'fast_searches': 100,
        'parallel_searches': {
            'kind': 'staged',
            'stages': [
                {'start_generation': 0, 'value': 1},
                {'start_generation': 30, 'value': 4},
            ],
        },
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'exploration_constant': 1.5,
        'first_play_urgency': {'kind': 'zero'},
        'forced_playouts': {'kind': 'disabled'},
    }


def test_parallel_searches_resolves_as_generation_staged_schedule() -> None:
    search = SelfPlaySearchParams.model_validate(_search_params_payload())

    assert tuple(search.parallel_searches.value_at(generation) for generation in (0, 29, 30, 100)) == (1, 1, 4, 4)


def test_constant_parallel_searches_still_resolves() -> None:
    payload = _search_params_payload()
    payload['parallel_searches'] = 2
    search = SelfPlaySearchParams.model_validate(payload)

    assert search.parallel_searches.value_at(0) == 2
    assert search.parallel_searches.value_at(500) == 2


def test_parallel_searches_must_stay_below_the_full_search_budget_at_every_stage() -> None:
    payload = _search_params_payload()
    payload['parallel_searches'] = {
        'kind': 'staged',
        'stages': [
            {'start_generation': 0, 'value': 1},
            {'start_generation': 20, 'value': 300},
        ],
    }

    with pytest.raises(ValidationError, match='must exceed the parallel-search count'):
        SelfPlaySearchParams.model_validate(payload)


def test_virtual_loss_weight_defaults_to_full_loss_and_rejects_out_of_range() -> None:
    search = SelfPlaySearchParams.model_validate(_search_params_payload())
    assert search.virtual_loss_weight == 1.0

    weighted_payload = _search_params_payload()
    weighted_payload['virtual_loss_weight'] = 0.5
    assert SelfPlaySearchParams.model_validate(weighted_payload).virtual_loss_weight == 0.5

    invalid_payload = _search_params_payload()
    invalid_payload['virtual_loss_weight'] = 1.5
    with pytest.raises(ValidationError):
        SelfPlaySearchParams.model_validate(invalid_payload)


def test_batched_inference_backend_is_typed_and_serialized() -> None:
    configuration = BatchedInferenceParams(
        inference_workers=2,
        inference_batch_size=64,
        outstanding_batches_per_worker=2,
        sdpa_backend=SdpaBackend.MEMORY_EFFICIENT,
    )

    assert configuration.sdpa_backend is SdpaBackend.MEMORY_EFFICIENT
    assert configuration.model_dump(mode='json')['sdpa_backend'] == 'memory_efficient'


def test_existing_inference_configuration_preserves_automatic_dispatch() -> None:
    configuration = BatchedInferenceParams(
        inference_workers=1,
        inference_batch_size=64,
        outstanding_batches_per_worker=1,
    )

    assert configuration.sdpa_backend is SdpaBackend.AUTOMATIC
    assert configuration.model_dump(exclude_unset=True) == configuration.model_dump()
