from __future__ import annotations

from pathlib import Path

from src.evaluation.configuration import EvaluationSearchConfiguration
from src.evaluation.match import SearchActionSelector, _create_opponent_selector
from src.self_play.configuration import BatchedInferenceParams
from src.training.checkpoint import CheckpointReference
from test_helpers.checkpoints import checkpoint_reference
from tools.distill_match import Arguments, _match_job, _student_action_selector


class SearchToken:
    pass


class RecordingGame:
    def __init__(self) -> None:
        self.calls: list[tuple[int, CheckpointReference, EvaluationSearchConfiguration]] = []

    def create_evaluation_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        configuration: EvaluationSearchConfiguration,
    ) -> SearchToken:
        self.calls.append((device_id, checkpoint, configuration))
        return SearchToken()


def _search(searches_per_move: int, parallel_searches: int) -> EvaluationSearchConfiguration:
    return EvaluationSearchConfiguration(
        searches_per_move=searches_per_move,
        parallel_searches=parallel_searches,
        exploration_constant=1.5,
        inference=BatchedInferenceParams(
            inference_workers=1,
            inference_batch_size=64,
            outstanding_batches_per_worker=1,
        ),
    )


def _arguments() -> Arguments:
    return Arguments(
        teacher_run_state=Path('teacher'),
        teacher_generation=94,
        student_run_state=Path('student'),
        student_generation=0,
        openings_manifest=Path('openings.json'),
        mode='equal-compute',
        searches_per_move=64,
        parallel_searches=4,
        exploration_constant=1.5,
        opening_pair_count=100,
        maximum_game_plies=300,
        bootstrap_samples=10_000,
        device_id=3,
        random_seed=7,
        output=Path('result.json'),
        experiment_config=Path('experiment.yaml'),
        pinned_throughput_ratio=3.5,
        throughput_position_count=200,
        throughput_duration_seconds=None,
    )


def test_distillation_match_wires_asymmetric_budgets_to_both_search_selectors() -> None:
    teacher = checkpoint_reference(generation=94)
    student = checkpoint_reference(generation=0)
    teacher_search = _search(64, 4)
    student_search = _search(224, 4)
    game = RecordingGame()
    job = _match_job(_arguments(), teacher, student, teacher_search)

    student_selector = _student_action_selector(game, 3, student, student_search)
    teacher_selector = _create_opponent_selector(job, game)

    assert isinstance(student_selector, SearchActionSelector)
    assert student_selector.searches_per_move == 224
    assert student_selector.parallel_searches == 4
    assert isinstance(teacher_selector, SearchActionSelector)
    assert teacher_selector.searches_per_move == 64
    assert teacher_selector.parallel_searches == 4
    assert game.calls == [(3, student, student_search), (3, teacher, teacher_search)]
