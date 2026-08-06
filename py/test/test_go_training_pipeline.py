from __future__ import annotations

from pathlib import Path

import pytest
import torch


AlphaZeroCpp = pytest.importorskip('AlphaZeroCpp')

if not hasattr(AlphaZeroCpp, 'GoBatchedSearch7'):
    pytest.skip('AlphaZeroCpp must be rebuilt with the R8 Go pipeline.', allow_module_level=True)

from src.experiment.chess_experiment import GoExperimentConfiguration, load_experiment_configuration
from src.games.go.contract import GoStateContract, GoSymmetryIndex
from src.games.go.training import calculate_go_loss, create_go_model
from src.self_play.chess_completed_game import SparseSearchVisit
from src.self_play.go_completed_game import (
    GoCompletedGame,
    GoCompletedGamePublisher,
    GoGameIdentity,
    GoMoveSelectionMode,
    GoRepresentationMetadata,
    GoRulesMetadata,
    GoSearchObservation,
    GoTerminationReason,
    go_completed_game_from_path,
)
from src.self_play.GoSelfPlay import GoSelfPlay
from src.self_play.value_target import FinalOutcome
from src.train.GoReplay import (
    GoReplayMaintainer,
    build_go_training_batch,
    materialize_go_game,
    rebuild_go_replay,
)


def _configuration() -> GoExperimentConfiguration:
    configuration = load_experiment_configuration(Path('configs/go-7x7-default-experiment.yaml'))
    assert isinstance(configuration, GoExperimentConfiguration)
    return configuration


def _two_pass_game(identity: GoGameIdentity = GoGameIdentity(run_id=1, worker_id=2, game_number=0)) -> GoCompletedGame:
    legal_actions = tuple(range(50))
    observations = tuple(
        GoSearchObservation(
            ply=ply,
            model_generation=3,
            legal_action_ids=legal_actions,
            visits=(SparseSearchVisit(action_id=49, visit_count=10),),
            root_value=-0.25 if ply == 0 else 0.25,
            selected_action_id=49,
            move_selection_mode=GoMoveSelectionMode.GREEDY,
            search_budget=10,
        )
        for ply in range(2)
    )
    return GoCompletedGame(
        identity=identity,
        rules=GoRulesMetadata(komi_half_points=15, maximum_moves=196),
        representation=GoRepresentationMetadata(board_size=7),
        model_generation=3,
        minimum_model_generation=3,
        created_at_seconds=100.0,
        generation_seconds=1.0,
        actions=(49, 49),
        final_current_player=1,
        final_score=-1.0,
        termination_reason=GoTerminationReason.TWO_PASSES,
        observations=observations,
    )


def test_go_encoding_targets_and_symmetries_match_deterministic_fixture() -> None:
    game = _two_pass_game()
    samples = materialize_go_game(game)
    contract = GoStateContract(7)

    assert len(samples) == 2
    assert samples[0].metadata.ply == 1
    assert samples[0].value_target.final_outcome is FinalOutcome.WIN
    assert samples[1].value_target.final_outcome is FinalOutcome.LOSS
    assert len(samples[1].encoded_state) == contract.packed_planes.payload_bytes
    assert contract.transform_action(0, GoSymmetryIndex.ROTATE_90) == 6
    assert contract.transform_action(0, GoSymmetryIndex.REFLECT_ROTATE_270) == 0
    assert contract.transform_action(contract.pass_action, GoSymmetryIndex.ROTATE_90) == contract.pass_action

    rotated = contract.transform_state(samples[1].encoded_state, GoSymmetryIndex.ROTATE_90)
    assert rotated == samples[1].encoded_state


def test_go_archive_rebuild_matches_live_ingestion(tmp_path: Path) -> None:
    publisher = GoCompletedGamePublisher(tmp_path, run_id=1, worker_id=2)
    publisher.publish(_two_pass_game(publisher.reserve_identity()))
    contract = GoStateContract(7)
    live = GoReplayMaintainer(tmp_path, contract, capacity=10, sampler_seed=7).maintain(10)
    rebuilt = rebuild_go_replay(tmp_path, contract, capacity=10, sampler_seed=7)

    assert live.samples == rebuilt.samples
    assert live.credited_samples == rebuilt.credited_samples == 2
    assert live.credited_completed_searches == rebuilt.credited_completed_searches == 20


def test_go_ddp_batch_and_optimizer_smoke() -> None:
    configuration = _configuration()
    game = _two_pass_game()
    samples = materialize_go_game(game)
    from src.train.GoReplay import GoReplaySnapshot

    snapshot = GoReplaySnapshot(
        contract=GoStateContract(7),
        samples=samples * 2,
        credited_samples=4,
        credited_completed_searches=40,
        sampler_seed=11,
        frozen_at_seconds=101.0,
        evicted_samples=0,
    )
    rank_zero = snapshot.rank_indices(0, 1, 4, 2, 0)
    rank_one = snapshot.rank_indices(0, 1, 4, 2, 1)
    assert set(rank_zero).isdisjoint(rank_one)
    batch = build_go_training_batch(snapshot, rank_zero, global_step=0, rank=0)
    assert batch.states.shape == (2, 17, 7, 7)
    assert torch.allclose(batch.policy_targets.sum(dim=1), torch.ones(2))

    torch.manual_seed(0)
    model = create_go_model(configuration, torch.device('cpu'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = calculate_go_loss(model, batch, configuration.go.objective)
    assert torch.isfinite(loss.total)
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()


def test_safety_cap_game_has_no_targets_or_credits() -> None:
    game = _two_pass_game().validated_copy(
        update={
            'final_score': None,
            'termination_reason': GoTerminationReason.MAXIMUM_MOVES.value,
            'observations': [
                observation.validated_copy(update={'sample_eligible': False}).model_dump(mode='json')
                for observation in _two_pass_game().observations
            ],
        }
    )
    assert materialize_go_game(game) == ()


class _PassModel(torch.nn.Module):
    def __init__(self, action_size: int, pass_action: int, outcome: tuple[float, float, float]) -> None:
        super().__init__()
        self.action_size = action_size
        self.pass_action = pass_action
        self.register_buffer('outcome', torch.tensor(outcome, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.zeros((states.shape[0], self.action_size), dtype=states.dtype, device=states.device)
        policy[:, self.pass_action] = 1.0
        return policy, self.outcome.to(dtype=states.dtype).expand(states.shape[0], -1)


def _write_pass_model(path: Path, outcome: tuple[float, float, float]) -> None:
    model = _PassModel(50, 49, outcome).eval()
    traced = torch.jit.trace(model, torch.zeros((1, 17, 7, 7), dtype=torch.float32))
    torch.jit.save(traced, path)


def test_cpu_go_self_play_publication_and_model_refresh_reset(tmp_path: Path) -> None:
    configuration = _configuration()
    search = configuration.training.self_play.search.validated_copy(
        update={'num_searches_per_turn': 4, 'num_parallel_searches': 1}
    )
    self_play_parameters = configuration.training.self_play.validated_copy(
        update={'search': search.model_dump(mode='json'), 'num_moves_after_which_to_play_greedy': 1}
    )
    topology = configuration.training.topology.validated_copy(
        update={
            'self_play': configuration.training.topology.self_play.validated_copy(
                update={'parallel_games_per_process': 1}
            ).model_dump(mode='json')
        }
    )
    training = configuration.training.validated_copy(
        update={
            'self_play': self_play_parameters.model_dump(mode='json'),
            'topology': topology.model_dump(mode='json'),
        }
    )
    configuration = configuration.validated_copy(update={'training': training.model_dump(mode='json')})
    initial_model = tmp_path / 'model-0.jit.pt'
    refreshed_model = tmp_path / 'model-1.jit.pt'
    _write_pass_model(initial_model, (0.2, 0.6, 0.2))
    _write_pass_model(refreshed_model, (0.6, 0.2, 0.2))
    publisher = GoCompletedGamePublisher(tmp_path, run_id=5, worker_id=0)
    self_play = GoSelfPlay(configuration, initial_model, 0, publisher, device_id=0)

    published = self_play.generate(1)
    completed = go_completed_game_from_path(published[0])
    assert completed.actions == (49, 49)
    assert completed.termination_reason is GoTerminationReason.TWO_PASSES

    root = self_play.search.new_root(self_play.rules)
    self_play.search.search([root], 4)
    assert root.visits > 0
    self_play.refresh_model(1, refreshed_model)
    assert root.visits == 0
    assert self_play.search.model_generation == 1
