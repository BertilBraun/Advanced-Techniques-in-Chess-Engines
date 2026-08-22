from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from src.evaluation.configuration import (
    EvaluationSearchConfiguration,
    PolicyRandomOpponentEvaluationDefinition,
    RandomOpponentEvaluationDefinition,
)
from src.evaluation.contracts import (
    CandidateOutcome,
    MatchEvaluationJob,
    OpeningLine,
    OpeningSuiteManifest,
    RandomOpponent,
)
from src.evaluation.match import run_match
from src.self_play.configuration import BatchedInferenceParams
from src.training.checkpoint import CheckpointReference
from test_helpers.checkpoints import checkpoint_reference
from test_helpers.fake_game_state import FakePosition, binary_choice_fake_game_state


@dataclass
class FakeRoot:
    position: FakePosition


@dataclass(frozen=True)
class FakeVisit:
    action_id: int
    visit_count: int


@dataclass(frozen=True)
class FakeResult:
    search_visits: tuple[FakeVisit, ...]


@dataclass(frozen=True)
class FakeBatch:
    results: tuple[FakeResult, ...]


class FakeSearch:
    def new_root(self, position: FakePosition) -> FakeRoot:
        return FakeRoot(position)

    def request(self, root: FakeRoot, full_search: bool) -> FakeRoot:
        assert full_search
        return root

    def search(self, requests: list[FakeRoot]) -> FakeBatch:
        return FakeBatch(tuple(FakeResult((FakeVisit(0, 10), FakeVisit(1, 5))) for _ in requests))


class OverrideSelector:
    def choose_actions(self, positions: tuple[FakePosition, ...]) -> tuple[int, ...]:
        return tuple(1 for _ in positions)


class FakeGame:
    state = binary_choice_fake_game_state()

    def create_evaluation_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        configuration: EvaluationSearchConfiguration,
    ) -> FakeSearch:
        return FakeSearch()


class FixedPolicyModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy = torch.tensor((0.1, 0.9), device=inputs.device).expand(inputs.shape[0], 2)
        value = torch.tensor((0.2, 0.6, 0.2), device=inputs.device).expand(inputs.shape[0], 3)
        correction = torch.zeros((inputs.shape[0], 1), device=inputs.device)
        return policy, value, correction


def _checkpoint() -> CheckpointReference:
    return checkpoint_reference()


def test_shared_match_swaps_players_and_aggregates_pairs() -> None:
    search = EvaluationSearchConfiguration(
        searches_per_move=8,
        parallel_searches=1,
        exploration_constant=1.0,
        inference=BatchedInferenceParams(
            inference_workers=1,
            inference_batch_size=8,
            outstanding_batches_per_worker=1,
        ),
    )
    definition = RandomOpponentEvaluationDefinition(
        kind='random',
        definition_id='random',
        opening_pair_count=1,
        search=search,
        maximum_game_plies=6,
    )
    job = MatchEvaluationJob(
        kind='match',
        job_id='job',
        definition=definition,
        boundary_seconds=1200,
        candidate=_checkpoint(),
        opponent=RandomOpponent(kind='random'),
        device_id=0,
        deadline_seconds=3600,
        random_seed=7,
        result_path=Path('result.json'),
    )
    opening = OpeningLine(
        opening_id='opening',
        action_ids=(0, 0, 0, 0),
        path_probability=0.5,
        final_position_digest='0' * 64,
        human_readable='opening',
    )
    openings = OpeningSuiteManifest(
        game='chess',
        rules_digest='1' * 64,
        representation_digest='2' * 64,
        random_seed=0,
        engine_identity='fake',
        engine_artifact_sha256=('3' * 64,),
        label_search_limit=10,
        expanded_actions_per_position=2,
        beam_width=2,
        openings=(opening,),
        builder_source_revision='revision',
    )

    result = run_match(job, FakeGame(), openings, 100, None, 'cpu')

    assert tuple(game.candidate_player for game in result.games) == ('first', 'second')
    assert all(game.outcome is CandidateOutcome.DRAW for game in result.games)
    assert result.aggregate.draws == 2
    assert result.aggregate.pair_count == 1
    assert result.aggregate.score == 0.5


def test_shared_match_accepts_candidate_selector_override() -> None:
    search = EvaluationSearchConfiguration(
        searches_per_move=8,
        parallel_searches=1,
        exploration_constant=1.0,
        inference=BatchedInferenceParams(
            inference_workers=1,
            inference_batch_size=8,
            outstanding_batches_per_worker=1,
        ),
    )
    definition = RandomOpponentEvaluationDefinition(
        kind='random',
        definition_id='random',
        opening_pair_count=1,
        search=search,
        maximum_game_plies=6,
    )
    job = MatchEvaluationJob(
        kind='match',
        job_id='override-job',
        definition=definition,
        boundary_seconds=1200,
        candidate=_checkpoint(),
        opponent=RandomOpponent(kind='random'),
        device_id=0,
        deadline_seconds=3600,
        random_seed=7,
        result_path=Path('result.json'),
    )
    openings = OpeningSuiteManifest(
        game='chess',
        rules_digest='1' * 64,
        representation_digest='2' * 64,
        random_seed=0,
        engine_identity='fake',
        engine_artifact_sha256=('3' * 64,),
        label_search_limit=10,
        expanded_actions_per_position=2,
        beam_width=2,
        openings=(
            OpeningLine(
                opening_id='opening',
                action_ids=(0, 0, 0, 0),
                path_probability=0.5,
                final_position_digest='0' * 64,
                human_readable='opening',
            ),
        ),
        builder_source_revision='revision',
    )

    result = run_match(
        job,
        FakeGame(),
        openings,
        100,
        None,
        'cpu',
        candidate_selector=OverrideSelector(),
    )

    assert result.games[0].played_action_ids[0] == 1
    assert result.games[1].played_action_ids[1] == 1


def test_policy_random_match_uses_direct_greedy_policy(tmp_path: Path) -> None:
    checkpoint = _checkpoint().model_copy(update={'inference_model_path': tmp_path / 'inference.pt'})
    traced = torch.jit.trace(FixedPolicyModel(), torch.zeros((1, 1, 1, 1)))
    torch.jit.save(traced, checkpoint.inference_model_path)
    definition = PolicyRandomOpponentEvaluationDefinition(
        kind='policy_random',
        definition_id='policy-random',
        opening_pair_count=1,
        maximum_game_plies=6,
    )
    job = MatchEvaluationJob(
        kind='match',
        job_id='policy-job',
        definition=definition,
        boundary_seconds=1200,
        candidate=checkpoint,
        opponent=RandomOpponent(kind='random'),
        device_id=0,
        deadline_seconds=3600,
        random_seed=7,
        result_path=tmp_path / 'result.json',
    )
    openings = OpeningSuiteManifest(
        game='chess',
        rules_digest='1' * 64,
        representation_digest='2' * 64,
        random_seed=0,
        engine_identity='fake',
        engine_artifact_sha256=('3' * 64,),
        label_search_limit=10,
        expanded_actions_per_position=2,
        beam_width=2,
        openings=(
            OpeningLine(
                opening_id='opening',
                action_ids=(0, 0, 0, 0),
                path_probability=0.5,
                final_position_digest='0' * 64,
                human_readable='opening',
            ),
        ),
        builder_source_revision='revision',
    )

    result = run_match(job, FakeGame(), openings, 100, None, 'cpu')

    assert result.games[0].played_action_ids[0] == 1
    assert result.games[1].played_action_ids[1] == 1
