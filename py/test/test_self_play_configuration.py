from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.self_play.configuration import BatchedInferenceParams, SdpaBackend, SelfPlaySearchParams


def _search_params_payload() -> dict[str, object]:
    return {
        'baseline_visits': {
            'kind': 'staged',
            'stages': [
                {'start_generation': 0, 'value': 200},
                {'start_generation': 30, 'value': 600},
            ],
        },
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'exploration_constant': 1.5,
        'first_play_urgency': {'kind': 'zero'},
        'forced_playouts': {'kind': 'disabled'},
    }


def test_baseline_visits_resolve_as_generation_staged_schedule() -> None:
    search = SelfPlaySearchParams.model_validate(_search_params_payload())

    assert tuple(search.baseline_visits.value_at(generation) for generation in (0, 29, 30, 100)) == (
        200,
        200,
        600,
        600,
    )


def test_constant_baseline_visits_resolve() -> None:
    payload = _search_params_payload()
    payload['baseline_visits'] = 300
    search = SelfPlaySearchParams.model_validate(payload)

    assert search.baseline_visits.value_at(0) == 300
    assert search.baseline_visits.value_at(500) == 300


def test_baseline_visits_must_remain_positive() -> None:
    payload = _search_params_payload()
    payload['baseline_visits'] = 0

    with pytest.raises(ValidationError, match='baseline visit budget'):
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
