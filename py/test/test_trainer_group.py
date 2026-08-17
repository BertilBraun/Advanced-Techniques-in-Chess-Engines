from pathlib import Path
from typing import cast

import torch
import pytest
from AlphaZeroCpp import GameSearchVisit
from torch import nn

from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.games.contracts import WdlTarget
from src.replay.contracts import ReplaySample, SparsePolicyTarget
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayDescription
from src.replay.store import ReplayStore
from src.training.checkpoint import CheckpointReference
from src.training.progress import TrainingProgress
from src.training.trainer import TrainerGroup
from src.training.trainer.contracts import TrainerQuantum, TrainerStartup
from src.training.checkpoint.persistence import create_optimizer, save_model_and_optimizer
from src.training.configuration import TrainingCompilation, TrainingPrecision
from src.training.network import Network
import src.training.trainer.rank as trainer_rank
from src.training.trainer.rank import DistributedTrainingModel


def _configuration(tmp_path: Path) -> ChessExperimentConfiguration:
    root = Path(__file__).parents[1]
    configuration = load_experiment_configuration(root / 'configs' / 'chess-experiment-template.yaml')
    trainer = configuration.training.trainer.validated_copy(
        update={
            'global_batch_size': 2,
            'local_batch_size': 2,
            'learning_rate': 0.001,
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
            'capacity': 2,
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
                        GameSearchVisit(action_id=0, visit_count=3),
                        GameSearchVisit(action_id=1, visit_count=1),
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
    trainer_group = TrainerGroup(
        configuration,
        game,
        TrainerStartup(
            network=configuration.training.network,
            save_path=tmp_path,
            starting_generation=starting_checkpoint.generation,
        ),
    )

    result = trainer_group.train_quantum(
        TrainerQuantum(
            replay=description,
            model_progress=TrainingProgress(
                completed_optimizer_steps=0,
                optimizer_steps_per_generation=configuration.training.lifecycle.credit.optimizer_steps_per_quantum,
            ),
            replay_source_progress=TrainingProgress(
                completed_optimizer_steps=0,
                optimizer_steps_per_generation=configuration.training.lifecycle.credit.optimizer_steps_per_quantum,
            ),
        )
    )
    trainer_group.close()

    assert result.completed_optimizer_steps == 1
    assert result.checkpoint.generation == 1
    assert result.checkpoint.manifest_path.is_file()
    assert result.statistics.elapsed_seconds > 0.0
    assert result.statistics.gradient_norm > 0.0
    assert result.statistics.replay_rows_per_second > 0.0
    assert result.statistics.training_samples_per_second > 0.0


def test_distributed_training_model_has_only_batch_norm_buffers() -> None:
    configuration = load_experiment_configuration(
        Path(__file__).parents[1] / 'configs' / 'chess-experiment-template.yaml'
    )
    game = ChessImplementation(configuration)
    model = DistributedTrainingModel(
        Network(configuration.training.network, torch.device('cpu'), game.network_dimensions)
    )

    buffer_names = tuple(name for name, _ in model.named_buffers())

    assert buffer_names
    assert all(name.endswith(('running_mean', 'running_var', 'num_batches_tracked')) for name in buffer_names)


def test_distributed_training_disables_per_forward_buffer_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = load_experiment_configuration(
        Path(__file__).parents[1] / 'configs' / 'chess-experiment-template.yaml'
    )
    game = ChessImplementation(configuration)
    model = Network(configuration.training.network, torch.device('cpu'), game.network_dimensions)

    class CapturedDistributedModel:
        def __init__(
            self,
            module: nn.Module,
            device_ids: list[int] | None,
            broadcast_buffers: bool,
        ) -> None:
            self.module = module
            self.device_ids = device_ids
            self.broadcast_buffers = broadcast_buffers

    monkeypatch.setattr(trainer_rank, 'DistributedDataParallel', CapturedDistributedModel)

    distributed_model = trainer_rank._create_distributed_model(
        model,
        configuration.training.topology.trainer,
        TrainingCompilation.DISABLED,
        device_id=0,
    )
    captured = cast(CapturedDistributedModel, distributed_model)

    assert captured.device_ids is None
    assert captured.broadcast_buffers is False


def test_distributed_training_compiles_only_the_training_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = load_experiment_configuration(
        Path(__file__).parents[1] / 'configs' / 'chess-experiment-template.yaml'
    )
    game = ChessImplementation(configuration)
    model = Network(configuration.training.network, torch.device('cpu'), game.network_dimensions)
    compiled_modules: list[nn.Module] = []

    def compile_model(module: nn.Module) -> nn.Module:
        compiled_modules.append(module)
        return module

    class CapturedDistributedModel:
        def __init__(
            self,
            module: nn.Module,
            device_ids: list[int] | None,
            broadcast_buffers: bool,
        ) -> None:
            self.module = module
            self.device_ids = device_ids
            self.broadcast_buffers = broadcast_buffers

    monkeypatch.setattr(trainer_rank.torch, 'compile', compile_model)
    monkeypatch.setattr(trainer_rank, 'DistributedDataParallel', CapturedDistributedModel)

    trainer_rank._create_distributed_model(
        model,
        configuration.training.topology.trainer,
        TrainingCompilation.DEFAULT,
        device_id=0,
    )

    assert len(compiled_modules) == 1
    assert isinstance(compiled_modules[0], DistributedTrainingModel)
    assert compiled_modules[0].model is model


def test_bfloat16_training_requires_cuda(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    trainer = configuration.training.trainer.validated_copy(update={'precision': TrainingPrecision.BFLOAT16})

    with pytest.raises(ValueError, match='Bfloat16 mixed-precision training requires CUDA'):
        configuration.training.validated_copy(update={'trainer': trainer.model_dump(mode='json')})
