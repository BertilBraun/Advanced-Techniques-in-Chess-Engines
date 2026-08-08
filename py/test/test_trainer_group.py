from pathlib import Path

import torch

from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.games.contracts import WdlTarget
from src.replay.contracts import ReplaySample, SparsePolicyTarget
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayDescription
from src.replay.store import ReplayStore
from src.self_play.completed_game import SparseSearchVisit
from src.training.checkpoint import CheckpointReference
from src.training.progress import TrainingProgress
from src.training.trainer_group import TrainerGroup
from src.util.save_paths import create_optimizer, save_model_and_optimizer
from src.neural_network import Network


def _configuration(tmp_path: Path) -> ChessExperimentConfiguration:
    root = Path(__file__).parents[1]
    configuration = load_experiment_configuration(root / 'configs' / 'chess-experiment-template.yaml')
    trainer = configuration.training.trainer.validated_copy(
        update={
            'global_batch_size': 2,
            'local_batch_size': 2,
            'learning_rate': {'kind': 'constant', 'value': 0.001},
        }
    )
    network = configuration.training.network.validated_copy(
        update={
            'num_layers': 1,
            'hidden_size': 8,
            'num_policy_channels': 2,
            'num_value_channels': 2,
            'value_fc_size': 8,
        }
    )
    topology = configuration.training.topology.validated_copy(
        update={
            'trainer': {
                'device_type': 'cpu',
                'process_group_backend': 'gloo',
                'rank_zero_device_id': 0,
                'ddp_device_ids': [0],
                'cpu_threads': 1,
                'interop_threads': 1,
            }
        }
    )
    credit = configuration.training.lifecycle.credit.validated_copy(
        update={
            'replay_ratio': 1,
            'optimizer_steps_per_quantum': 1,
            'maximum_optimizer_steps': 1,
        }
    )
    replay = configuration.training.lifecycle.replay.validated_copy(
        update={
            'capacity': {'kind': 'constant', 'value': 2},
            'maximum_capacity': 2,
            'maximum_policy_entries': 2,
        }
    )
    lifecycle = configuration.training.lifecycle.validated_copy(
        update={
            'credit': credit.model_dump(mode='json'),
            'replay': replay.model_dump(mode='json'),
        }
    )
    training = configuration.training.validated_copy(
        update={
            'save_path': str(tmp_path),
            'random_seed': 2,
            'network': network.model_dump(mode='json'),
            'trainer': trainer.model_dump(mode='json'),
            'topology': topology.model_dump(mode='json'),
            'lifecycle': lifecycle.model_dump(mode='json'),
        }
    )
    return configuration.validated_copy(update={'training': training.model_dump(mode='json')})


def test_trainer_group_runs_blocking_world_size_one_ddp_quantum(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    game = ChessImplementation(configuration)
    model = Network(configuration.training.network, torch.device('cpu'), game.network_dimensions)
    save_model_and_optimizer(model, create_optimizer(model, configuration.training.trainer.optimizer), 0, tmp_path)
    starting_checkpoint = CheckpointReference.load(tmp_path, 0)
    layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=2,
    )
    replay_path = tmp_path / 'replay.bin'
    store = ReplayStore.create(replay_path, layout, maximum_capacity=2, logical_capacity=2)
    for weight in (1.0, 2.0):
        store.append(
            ReplaySample(
                encoded_state=game.state.packed_plane_layout.value(bytes(game.state.packed_plane_layout.payload_bytes)),
                policy=SparsePolicyTarget(
                    visits=(
                        SparseSearchVisit(action_id=0, visit_count=3),
                        SparseSearchVisit(action_id=1, visit_count=1),
                    )
                ),
                wdl_target=WdlTarget(win=0.0, draw=1.0, loss=0.0),
                root_value=0.0,
                auxiliary_targets=(),
                sample_weight=weight,
                source_model_generation=0,
                source_created_at_seconds=1.0,
            )
        )
    store.flush()
    replay_state = store.state
    description = ReplayDescription(
        path=replay_path,
        head=replay_state.head,
        size=replay_state.size,
        logical_capacity=replay_state.logical_capacity,
        maximum_capacity=replay_state.maximum_capacity,
        layout=layout,
    )
    store.close()
    trainer_group = TrainerGroup(configuration, game, starting_checkpoint)

    result = trainer_group.train_quantum(
        description,
        TrainingProgress(
            completed_optimizer_steps=0,
            optimizer_steps_per_generation=configuration.training.lifecycle.credit.optimizer_steps_per_quantum,
        ),
    )
    trainer_group.close()

    assert result.completed_optimizer_steps == 1
    assert result.checkpoint.generation == 1
    assert result.checkpoint.manifest_path.is_file()
    assert result.statistics.elapsed_seconds > 0.0
    assert result.statistics.gradient_norm > 0.0
    assert result.statistics.replay_rows_per_second > 0.0
    assert result.statistics.training_samples_per_second > 0.0
