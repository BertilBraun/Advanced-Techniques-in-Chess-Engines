import os
from pathlib import Path

import pytest
import torch

from src.experiment.configuration import ExperimentConfiguration, load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.composition import create_game_implementation
from src.games.go.configuration import GoExperimentConfiguration
from src.replay.batch_loader import MappedReplayBatchLoader
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayManager
from src.self_play.worker import SelfPlayWorker
from src.training.checkpoint import CheckpointReference
from src.training.progress import TrainingProgress
from src.training.trainer import TrainerGroup
from src.training.checkpoint.persistence import create_model, create_optimizer, save_model_and_optimizer


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get('PHASE2_RUNTIME_SMOKE') != '1', reason='Phase 2 runtime smoke is opt-in.'),
]


def _tiny_configuration(path: Path, output_path: Path) -> ExperimentConfiguration:
    configuration = load_experiment_configuration(path)
    network = configuration.training.network.validated_copy(
        update={
            'num_layers': 1,
            'hidden_size': 8,
            'num_policy_channels': 2,
            'num_value_channels': 2,
            'value_fc_size': 8,
        }
    )
    trainer = configuration.training.trainer.validated_copy(
        update={
            'global_batch_size': 1,
            'local_batch_size': 1,
            'learning_rate': {'kind': 'constant', 'value': 0.001},
        }
    )
    trainer_topology = configuration.training.topology.trainer.validated_copy(
        update={
            'device_type': 'cpu',
            'process_group_backend': 'gloo',
            'rank_zero_device_id': 0,
            'ddp_device_ids': [0],
            'cpu_threads': 1,
            'interop_threads': 1,
        }
    )
    self_play_topology = configuration.training.topology.self_play.validated_copy(
        update={'device_ids': [0], 'parallel_games_per_process': 1}
    )
    topology = configuration.training.topology.validated_copy(
        update={
            'trainer': trainer_topology.model_dump(mode='json'),
            'self_play': self_play_topology.model_dump(mode='json'),
        }
    )
    replay = configuration.training.lifecycle.replay.validated_copy(
        update={
            'capacity': {'kind': 'constant', 'value': 128},
            'maximum_capacity': 128,
            'maximum_policy_entries': min(8, configuration.network_dimensions.actions),
        }
    )
    credit = configuration.training.lifecycle.credit.validated_copy(
        update={
            'replay_ratio': 1,
            'optimizer_steps_per_quantum': 1,
            'maximum_optimizer_steps': 1,
        }
    )
    lifecycle = configuration.training.lifecycle.validated_copy(
        update={
            'replay': replay.model_dump(mode='json'),
            'credit': credit.model_dump(mode='json'),
        }
    )
    training = configuration.training.validated_copy(
        update={
            'save_path': str(output_path),
            'network': network.model_dump(mode='json'),
            'trainer': trainer.model_dump(mode='json'),
            'topology': topology.model_dump(mode='json'),
            'lifecycle': lifecycle.model_dump(mode='json'),
        }
    )
    common_self_play = {
        'search': {
            'full_searches': {'kind': 'constant', 'value': 2},
            'fast_searches': {'kind': 'constant', 'value': 1},
            'parallel_searches': 1,
            'dirichlet_epsilon': {'kind': 'constant', 'value': 0.0},
            'dirichlet_alpha': {'kind': 'constant', 'value': 0.3},
            'exploration_constant': {'kind': 'constant', 'value': 1.0},
            'first_play_urgency': {'kind': 'zero'},
            'forced_playouts': {'kind': 'disabled'},
        },
        'inference': {
            'inference_workers': 1,
            'inference_batch_size': 1,
            'outstanding_batches_per_worker': 1,
        },
        'start_position': {'kind': 'random_opening', 'maximum_plies': {'kind': 'constant', 'value': 0}},
        'full_search_probability': {'kind': 'constant', 'value': 1.0},
        'retained_root_visit_fraction': {'kind': 'constant', 'value': 0.0},
        'greedy_after_ply': {'kind': 'constant', 'value': 1},
        'starting_temperature': {'kind': 'constant', 'value': 1.0},
        'final_temperature': {'kind': 'constant', 'value': 1.0},
        'primary_sample_weight': {'kind': 'constant', 'value': 1.0},
        'detailed_statistics_workers': 1,
    }
    match configuration:
        case ChessExperimentConfiguration():
            chess = configuration.chess.validated_copy(
                update={
                    'self_play': {
                        **common_self_play,
                        'maximum_game_plies': {'kind': 'constant', 'value': 1},
                    }
                }
            )
            return configuration.validated_copy(
                update={'training': training.model_dump(mode='json'), 'chess': chess.model_dump(mode='json')}
            )
        case GoExperimentConfiguration():
            rules = configuration.go.rules.validated_copy(update={'maximum_moves': 98})
            go = configuration.go.validated_copy(
                update={'rules': rules.model_dump(mode='json'), 'self_play': common_self_play}
            )
            return configuration.validated_copy(
                update={'training': training.model_dump(mode='json'), 'go': go.model_dump(mode='json')}
            )


@pytest.mark.parametrize(
    'configuration_name',
    ('chess-experiment-template.yaml', 'go-7x7-experiment-template.yaml'),
)
def test_complete_phase2_cpu_path(configuration_name: str, tmp_path: Path) -> None:
    configuration_path = Path(__file__).parents[2] / 'configs' / configuration_name
    configuration = _tiny_configuration(configuration_path, tmp_path)
    game = create_game_implementation(configuration)
    device = torch.device('cpu')
    model = create_model(
        configuration.training.network,
        device,
        game.network_dimensions,
        tuple(head.action_size for head in game.target_layout.auxiliary_heads),
    )
    optimizer = create_optimizer(model, configuration.training.trainer.optimizer)
    save_model_and_optimizer(model, optimizer, 0, tmp_path)
    checkpoint = CheckpointReference.load(tmp_path, 0)
    inbox_path = tmp_path / 'completed-games' / 'inbox'
    worker = SelfPlayWorker(game, 1, 0, 0, inbox_path)
    worker.refresh_published_model(checkpoint)
    maximum_batches = 2 if configuration.game == 'chess' else 100
    for _ in range(maximum_batches):
        worker.run_batch()
        if tuple(inbox_path.glob('*.json')):
            break
    else:
        raise AssertionError('CPU self-play did not complete a game within its configured safety cap.')

    replay_layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=configuration.training.lifecycle.replay.maximum_policy_entries,
    )
    replay_manager = ReplayManager.open(
        tmp_path,
        game.state,
        replay_layout,
        configuration.training.lifecycle.replay,
        model_generation=0,
        value_discount_per_ply=game.value_discount_per_ply,
    )
    ingestion = replay_manager.ingest_available_games(0)
    description = replay_manager.description()
    replay_manager.close()
    assert ingestion.games_ingested == 1
    assert ingestion.samples_added >= 1

    batch = next(
        iter(
            MappedReplayBatchLoader(
                replay=description,
                state=game.state,
                source_optimizer_step=0,
                optimizer_steps=1,
                global_batch_size=1,
                world_size=1,
                rank=0,
                sampler_seed=0,
                pin_memory=False,
            )
        )
    )
    assert batch.states.shape[0] == 1

    trainer_group = TrainerGroup(configuration, game, checkpoint)
    result = trainer_group.train_quantum(
        description,
        TrainingProgress(
            completed_optimizer_steps=0,
            optimizer_steps_per_generation=configuration.training.lifecycle.credit.optimizer_steps_per_quantum,
        ),
    )
    trainer_group.close()
    worker.refresh_published_model(result.checkpoint)

    assert result.checkpoint.generation == 1
    assert worker.model_generation == 1
    assert result.statistics.training_samples_per_second > 0.0
    print(
        f'{configuration.game}: ingestion={ingestion.samples_per_second:.0f} samples/s, '
        f'replay={result.statistics.replay_rows_per_second:.0f} rows/s, '
        f'ddp={result.statistics.training_samples_per_second:.1f} samples/s'
    )
