from __future__ import annotations

from src.search_budget.sampling import LabelPositionIdentity, partition_generation_sample, select_generation_sample


def identities(count: int) -> tuple[LabelPositionIdentity, ...]:
    return tuple(
        LabelPositionIdentity(source_generation=7, game_identity=f'worker:instance:{index // 10}', ply=index % 10)
        for index in range(count)
    )


def test_sampling_selects_exact_floor_two_percent_without_replacement() -> None:
    selected = select_generation_sample(identities(149), run_seed=20260827)
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_sampling_is_stable_under_input_order_and_seeded() -> None:
    positions = identities(500)
    selected = select_generation_sample(positions, run_seed=42)
    assert select_generation_sample(tuple(reversed(positions)), run_seed=42) == selected
    assert select_generation_sample(positions, run_seed=43) != selected


def test_sampling_rejects_duplicate_or_cross_generation_identities() -> None:
    duplicate = identities(1)[0]
    try:
        select_generation_sample((duplicate, duplicate), run_seed=1)
    except ValueError as error:
        assert 'unique' in str(error)
    else:
        raise AssertionError('Duplicate identities were accepted.')

    other_generation = duplicate.model_copy(update={'source_generation': 8})
    try:
        select_generation_sample((duplicate, other_generation), run_seed=1)
    except ValueError as error:
        assert 'source generations' in str(error)
    else:
        raise AssertionError('Cross-generation identities were accepted.')


def test_generation_sample_uses_fixed_512_position_shards_and_one_remainder() -> None:
    shards = partition_generation_sample(identities(1200))
    assert tuple(len(shard) for shard in shards) == (512, 512, 176)
    assert tuple(position for shard in shards for position in shard) == identities(1200)
