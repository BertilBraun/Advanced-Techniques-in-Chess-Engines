import random

from tools.benchmark_repetition_mcts import (
    CacheCounters,
    UniqueRandomStartGenerator,
    select_full_searches,
    starting_position_digest,
    subtract_cache_counters,
)


def test_random_starts_are_unique_and_deterministic() -> None:
    first = UniqueRandomStartGenerator(random_seed=17, maximum_opening_plies=12)
    second = UniqueRandomStartGenerator(random_seed=17, maximum_opening_plies=12)

    first_starts = tuple(first.next() for _ in range(256))
    second_starts = tuple(second.next() for _ in range(256))

    first_encodings = tuple(start.encoding for start in first_starts)
    assert len(set(first_encodings)) == 256
    assert first_encodings == tuple(start.encoding for start in second_starts)
    assert all(1 <= start.opening_plies <= 12 for start in first_starts)
    assert all(not start.position.is_terminal for start in first_starts)


def test_starting_position_digest_preserves_order() -> None:
    first = b'first'
    second = b'second'

    assert starting_position_digest((first, second)) == starting_position_digest((first, second))
    assert starting_position_digest((first, second)) != starting_position_digest((second, first))


def test_cache_counter_delta_preserves_cumulative_set_size() -> None:
    before = CacheCounters(
        total_positions=10,
        unique_hashes=8,
        repeated_hashes=2,
        same_batch_repeats=1,
        prior_batch_repeats=1,
        set_size=8,
    )
    after = CacheCounters(
        total_positions=25,
        unique_hashes=18,
        repeated_hashes=7,
        same_batch_repeats=3,
        prior_batch_repeats=4,
        set_size=18,
    )

    assert subtract_cache_counters(after, before) == CacheCounters(
        total_positions=15,
        unique_hashes=10,
        repeated_hashes=5,
        same_batch_repeats=2,
        prior_batch_repeats=3,
        set_size=18,
    )


def test_mixed_search_schedule_is_deterministic() -> None:
    first = select_full_searches(512, 0.25, random.Random(23))
    second = select_full_searches(512, 0.25, random.Random(23))

    assert first == second
    assert 100 < sum(first) < 160
    assert len(first) - sum(first) > 350
