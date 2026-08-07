from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import src.training.replay as replay_module


AlphaZeroCpp = pytest.importorskip('AlphaZeroCpp')

if not hasattr(AlphaZeroCpp, 'GoSelfPlaySearch7'):
    pytest.skip('AlphaZeroCpp must be rebuilt with the R8 Go pipeline.', allow_module_level=True)

import src.games.go.replay as go_replay_module

from src.experiment.configuration import load_experiment_configuration
from src.games.go.configuration import GoExperimentConfiguration
from src.experiment.run import ExperimentRunManifest, experiment_sha256
from src.experiment.run_contract import ApprovalRecord, ResolvedHardware
from src.games.go.contract import GoStateContract, GoSymmetryIndex
from src.games.go.training import GoImplementation, calculate_go_loss, create_go_model
from src.training.trainer_process import TrainerProcess
from src.self_play.completed_game import CompletedGamePublisher, GameIdentity, SparseSearchVisit
from src.self_play.completed_game_record import completed_game_from_path
from src.games.go.completed_game import (
    GoCompletedGame,
    GoMoveSelectionMode,
    GoRepresentationMetadata,
    GoRulesMetadata,
    GoSearchObservation,
    GoTerminationReason,
)
from src.games.go.self_play import GoSelfPlayPolicy
from src.self_play.worker import SelfPlayWorker
from src.self_play.value_target import FinalOutcome
from src.games.go.replay import (
    GoReplayImplementation,
    build_go_training_batch,
    materialize_go_game,
    pack_go_visits,
    rebuild_go_replay,
)
from src.training.replay import ReplayMaintainer, ReplaySnapshot
from src.util.save_paths import create_optimizer, save_model_and_optimizer


def _configuration() -> GoExperimentConfiguration:
    configuration = load_experiment_configuration(Path('configs/go-7x7-experiment-template.yaml'))
    assert isinstance(configuration, GoExperimentConfiguration)
    return configuration


def _two_pass_game(identity: GameIdentity = GameIdentity(run_id=1, worker_id=2, game_number=0)) -> GoCompletedGame:
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


@pytest.mark.parametrize('symmetry', tuple(GoSymmetryIndex))
def test_go_batch_applies_one_selected_symmetry_to_state_and_policy(
    symmetry: GoSymmetryIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = GoStateContract(7)
    rules = AlphaZeroCpp.GoRules(komi_half_points=15, maximum_moves=196)
    encoded_state = contract.packed_position(contract.initial_position(rules).child(0))
    sample = replace(
        materialize_go_game(_two_pass_game())[0],
        encoded_state=encoded_state,
        visits=pack_go_visits(((0, 10),), contract.action_size),
    )
    snapshot = ReplaySnapshot(
        samples=(sample,),
        credited_samples=1,
        credited_completed_searches=10,
        sampler_seed=11,
        frozen_at_seconds=101.0,
        evicted_samples=0,
        estimated_sample_bytes=0,
        encoded_state_value_overhead_bytes=0,
        projected_capacity_bytes=0,
        projected_review_capacity_bytes=0,
    )

    def selected_symmetry(
        sampler_seed: int,
        global_step: int,
        rank: int,
        sample_position: int,
    ) -> GoSymmetryIndex:
        del sampler_seed, global_step, rank, sample_position
        return symmetry

    monkeypatch.setattr(go_replay_module, '_symmetry_index', selected_symmetry)
    batch = build_go_training_batch(contract, snapshot, (0,), global_step=4, rank=0)

    expected_states = torch.empty_like(batch.states).numpy()
    contract.decode_batch_into((contract.transform_state(encoded_state, symmetry),), expected_states)
    torch.testing.assert_close(batch.states, torch.from_numpy(expected_states))
    assert batch.policy_targets.argmax(dim=1).item() == contract.transform_action(0, symmetry)


def test_go_archive_rebuild_matches_live_ingestion(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, run_id=1, worker_id=2)
    publisher.publish(_two_pass_game(publisher.reserve_identity()))
    contract = GoStateContract(7)
    live, _ = ReplayMaintainer(
        tmp_path,
        GoReplayImplementation(contract),
        capacity=10,
        sampler_seed=7,
    ).maintain(10)
    rebuilt = rebuild_go_replay(tmp_path, contract, capacity=10, sampler_seed=7)

    assert live.samples == rebuilt.samples
    assert live.credited_samples == rebuilt.credited_samples == 2
    assert live.credited_completed_searches == rebuilt.credited_completed_searches == 20


def test_go_archive_restart_materializes_only_capacity_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = CompletedGamePublisher(tmp_path, run_id=3, worker_id=0)
    for _ in range(5):
        publisher.publish(_two_pass_game(publisher.reserve_identity()))
    contract = GoStateContract(7)
    implementation = GoReplayImplementation(contract)
    ReplayMaintainer(tmp_path, implementation, capacity=20, sampler_seed=13).maintain(20)
    original_reader = replay_module.read_frame_payload
    read_sequences: list[int] = []

    def track_read(frame_index: replay_module.ArchiveFrameIndex) -> bytes:
        read_sequences.append(frame_index.ingestion_sequence)
        return original_reader(frame_index)

    monkeypatch.setattr(replay_module, 'read_frame_payload', track_read)
    snapshot, _ = ReplayMaintainer(tmp_path, implementation, capacity=3, sampler_seed=13).maintain(3)

    assert read_sequences == [3, 4]
    assert len(snapshot.samples) == 3
    assert snapshot.credited_samples == 10


def test_go_ddp_batch_and_optimizer_smoke() -> None:
    configuration = _configuration()
    game = _two_pass_game()
    samples = materialize_go_game(game)
    snapshot = ReplaySnapshot(
        samples=samples * 2,
        credited_samples=4,
        credited_completed_searches=40,
        sampler_seed=11,
        frozen_at_seconds=101.0,
        evicted_samples=0,
        estimated_sample_bytes=0,
        encoded_state_value_overhead_bytes=0,
        projected_capacity_bytes=0,
        projected_review_capacity_bytes=0,
    )
    rank_zero = snapshot.rank_indices(0, 1, 4, 2, 0)
    rank_one = snapshot.rank_indices(0, 1, 4, 2, 1)
    assert set(rank_zero).isdisjoint(rank_one)
    batch = build_go_training_batch(GoStateContract(7), snapshot, rank_zero, global_step=0, rank=0)
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


def test_safety_cap_game_requires_a_scored_result() -> None:
    game = _two_pass_game().validated_copy(
        update={
            'termination_reason': GoTerminationReason.MAXIMUM_MOVES.value,
        }
    )
    assert game.final_score is not None
    assert all(observation.sample_eligible for observation in game.observations)


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


def test_go_uses_shared_policy_and_counted_analysis(tmp_path: Path) -> None:
    model_path = tmp_path / 'analysis.jit.pt'
    _write_pass_model(model_path, (0.6, 0.2, 0.2))
    analysis = AlphaZeroCpp.GoAnalysis7(
        AlphaZeroCpp.InferenceConfiguration(
            device_id=0,
            model_path=str(model_path),
            device=AlphaZeroCpp.InferenceDevice.CPU,
        ),
        AlphaZeroCpp.AnalysisParameters(
            parallel_searches=2,
            exploration_constant=1.0,
            inference=AlphaZeroCpp.BatchedInferenceParameters(1, 4, 2),
        ),
    )
    root = analysis.new_root(AlphaZeroCpp.GoRules(15, 196))

    policy = analysis.analyze_policy(root)
    searched = analysis.analyze_counted(root, 4)

    assert policy.chosen_action_id == 49
    assert policy.searches == 0
    assert searched.chosen_action_id == 49
    assert searched.searches == 4
    assert searched.principal_variation


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
    publisher = CompletedGamePublisher(tmp_path, run_id=5, worker_id=0)
    policy = GoSelfPlayPolicy(configuration, publisher, device_id=0)
    self_play = SelfPlayWorker(policy, parallel_game_count=1)
    self_play.refresh_published_model(0, initial_model)

    while not tuple(publisher.inbox_path.glob('*.json')):
        self_play.run_batch()
    published = tuple(publisher.inbox_path.glob('*.json'))
    completed = completed_game_from_path(published[0])
    assert isinstance(completed, GoCompletedGame)
    assert completed.actions == (49, 49)
    assert completed.termination_reason is GoTerminationReason.TWO_PASSES

    assert policy.search is not None
    root = policy.search.new_root(policy.rules)
    policy.search.search([policy.search_request_type(root, True)])
    assert root.visits > 0
    self_play.refresh_published_model(1, refreshed_model)
    assert root.visits == 0
    assert policy.search.model_generation == 1


def _smoke_configuration(tmp_path: Path) -> GoExperimentConfiguration:
    configuration = _configuration()
    search = configuration.training.self_play.search.validated_copy(
        update={'num_searches_per_turn': 4, 'num_parallel_searches': 1}
    )
    inference = configuration.training.self_play.inference.validated_copy(update={'inference_batch_size': 2})
    self_play_parameters = configuration.training.self_play.validated_copy(
        update={
            'search': search.model_dump(mode='json'),
            'inference': inference.model_dump(mode='json'),
            'num_moves_after_which_to_play_greedy': 1,
        }
    )
    trainer = configuration.training.trainer.validated_copy(
        update={
            'global_batch_size': 2,
            'local_batch_size': 2,
            'learning_rate': configuration.training.trainer.learning_rate.validated_copy(
                update={'optimizer_steps_per_model_version': 1}
            ).model_dump(mode='json'),
        }
    )
    topology = configuration.training.topology.validated_copy(
        update={
            'self_play': configuration.training.topology.self_play.validated_copy(
                update={'parallel_games_per_process': 1}
            ).model_dump(mode='json')
        }
    )
    credit = configuration.training.lifecycle.credit.validated_copy(
        update={
            'replay_ratio': 1,
            'optimizer_steps_per_quantum': 1,
            'maximum_optimizer_steps': 1,
            'initial_replay_capacity_unique_positions': 10,
            'maximum_replay_capacity_unique_positions': 10,
            'replay_capacity_ramp_model_versions': 1,
            'retained_checkpoint_interval_steps': 1,
        }
    )
    evaluation = configuration.training.lifecycle.evaluation.validated_copy(
        update={'interval_optimizer_steps': 1, 'full_interval_optimizer_steps': 1}
    )
    lifecycle = configuration.training.lifecycle.validated_copy(
        update={
            'credit': credit.model_dump(mode='json'),
            'evaluation': evaluation.model_dump(mode='json'),
        }
    )
    training = configuration.training.validated_copy(
        update={
            'save_path': str(tmp_path),
            'self_play': self_play_parameters.model_dump(mode='json'),
            'trainer': trainer.model_dump(mode='json'),
            'topology': topology.model_dump(mode='json'),
            'lifecycle': lifecycle.model_dump(mode='json'),
        }
    )
    return configuration.validated_copy(update={'training': training.model_dump(mode='json')})


def _write_go_run_manifest(run_path: Path, configuration: GoExperimentConfiguration) -> None:
    manifest = ExperimentRunManifest(
        experiment=configuration,
        approval=ApprovalRecord(
            approved_by='test',
            approved_at_utc=datetime.now(timezone.utc),
            run_name=configuration.run.run_name,
            source_revision='a' * 40,
            configuration_sha256=experiment_sha256(configuration),
            provider_name=configuration.run.hardware.provider_name,
            offer_id=configuration.run.hardware.offer_id,
            cost_currency=configuration.training.limits.cost_currency,
            hourly_price=configuration.training.limits.hourly_price,
            maximum_cost=None,
            maximum_wall_time_minutes=360,
        ),
        resolved_hardware=ResolvedHardware(
            visible_gpu_names=(),
            visible_gpu_count=0,
            logical_cpu_count=1,
            total_ram_gib=8,
            free_disk_gib=8,
        ),
        source_revision='a' * 40,
        source_worktree_clean=True,
        initial_model_sha256='b' * 64,
        evaluation_dataset_sha256=None,
        stockfish_binary_sha256=None,
        open_file_soft_limit=1024,
        torch_version=torch.__version__,
        cuda_version=None,
    )
    (run_path / 'run_manifest.json').write_text(manifest.model_dump_json(indent=2) + '\n', encoding='utf-8')


def test_go_shared_trainer_process_publishes_checkpoint(tmp_path: Path) -> None:
    configuration = _smoke_configuration(tmp_path)
    model = create_go_model(configuration, torch.device('cpu'))
    policy_output = model.policyHead[-1]
    assert isinstance(policy_output, torch.nn.Linear)
    with torch.no_grad():
        policy_output.weight.zero_()
        policy_output.bias.fill_(-20.0)
        policy_output.bias[49] = 20.0
    optimizer = create_optimizer(model, configuration.training.trainer.optimizer)
    save_model_and_optimizer(model, optimizer, 0, tmp_path)
    _write_go_run_manifest(tmp_path, configuration)

    publisher = CompletedGamePublisher(tmp_path, run_id=9, worker_id=0)
    publisher.publish(_two_pass_game(publisher.reserve_identity()))
    trainer = TrainerProcess(GoImplementation(configuration), run_id=9, starting_model_version=0)
    try:
        replay_state = trainer.maintain_replay(10)
        assert replay_state.credited_unique_samples == 2
        result = trainer.train_quantum(global_step=0, model_version=1)
        assert result.model_version == 1
        assert result.global_step == 1
    finally:
        trainer.close()
    assert (tmp_path / 'checkpoint_1.json').is_file()
