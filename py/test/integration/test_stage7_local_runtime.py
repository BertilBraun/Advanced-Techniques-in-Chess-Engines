from __future__ import annotations

import multiprocessing
from pathlib import Path
from uuid import UUID

import pytest
import torch
from src.az.config.base import DeterminismMode
from src.az.config.model import FixedModelSchedule
from src.az.config.resolution import resolve_configuration
from src.az.config.root import ResolvedRunConfiguration
from src.az.config.serialization import load_authoring_configuration
from src.az.config.runtime import DeviceAssignment
from src.az.config.search import FixedSearchBudget

native = pytest.importorskip('az_go_native', reason='focused native Go extension has not been built')

from src.az.config.training import (
    AdamWOptimizerConfiguration,
    ConstantLearningRate,
    ReplayCreditConfiguration,
    TrainingConfiguration,
)
from src.az.games.api import GameIdentifier
from src.az.games.go.module import create_go_training_module
from src.az.replay.credits import ReplayCreditJournal
from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ReplayShardStorage, ShardMetadata
from src.az.runtime.messages import WorkerModelRefreshed, WorkerProgress, WorkerResourceSample, WorkerStopped
from src.az.runtime.factory import RuntimeBuildEnvironment, build_runtime_plan
from src.az.runtime.orchestrator import RuntimeOrchestrator
from src.az.self_play.worker import run_go_worker
from src.az.training.checkpoints import CheckpointRepository
from src.az.training.distributed import TrainingRank
from src.az.training.trainer import CreditTrainer
from test.unit.go_stage5_helpers import game_configuration, model_configuration, objective_configuration

pytestmark = pytest.mark.integration
RUN_ID = UUID(int=701)
CONFIGURATION_SHA256 = 'c' * 64


def _runtime_configuration() -> ResolvedRunConfiguration:
    configuration = resolve_configuration(load_authoring_configuration(Path('configs/v2/go-7x7-fixed.authoring.json')))
    game = game_configuration().model_copy(update={'safety_ply_cap': 49})
    architecture = model_configuration().model_copy(
        update={'channels': 4, 'residual_blocks': 1, 'value_hidden_size': 8}
    )
    topology = configuration.topology.model_copy(
        update={
            'trainer': DeviceAssignment(device_ids=(0,)),
            'self_play': DeviceAssignment(device_ids=(0,)),
            'evaluation': DeviceAssignment(device_ids=(0,)),
            'self_play_workers_per_device': 1,
            'maximum_active_searches_per_worker': 2,
            'inference_workers_per_device': 1,
            'inference_batch_size': 2,
            'maximum_pending_inference_batches': 2,
        }
    )
    search = configuration.search.model_copy(
        update={
            'budget': FixedSearchBudget(kind='fixed', simulations=1),
            'inference': configuration.search.inference.model_copy(
                update={
                    'maximum_batch_size': 2,
                    'maximum_wait_microseconds': 100_000,
                    'cache_capacity': 32,
                }
            ),
        }
    )
    self_play = configuration.self_play.model_copy(update={'games_per_shard': 2})
    replay = configuration.replay.model_copy(update={'maximum_positions_per_shard': 98, 'capacity_positions': 512})
    training = configuration.training.model_copy(update={'global_batch_size': 1, 'local_batch_size': 1})
    experiment = configuration.experiment.model_copy(update={'duration_seconds': 20, 'checkpoint_elapsed_seconds': ()})
    model = configuration.model.model_copy(
        update={'schedule': FixedModelSchedule(kind='fixed', architecture=architecture)}
    )
    return ResolvedRunConfiguration.model_validate(
        configuration.model_copy(
            update={
                'experiment': experiment,
                'topology': topology,
                'game': game,
                'model': model,
                'search': search,
                'self_play': self_play,
                'replay': replay,
                'training': training,
                'evaluation': configuration.evaluation.model_copy(update={'checkpoint_elapsed_seconds': ()}),
                'telemetry': configuration.telemetry.model_copy(
                    update={'write_every_seconds': 1, 'resource_sample_every_seconds': 1}
                ),
            }
        ).model_dump()
    )


def test_bounded_multiprocess_runtime_generates_trains_refreshes_and_stops(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration()
    game = configuration.game
    model = configuration.model.schedule.architecture
    checkpoint_directory = (tmp_path / 'checkpoints').resolve()
    plan = build_runtime_plan(
        configuration,
        RuntimeBuildEnvironment(
            run_id=RUN_ID,
            resolved_configuration_sha256=CONFIGURATION_SHA256,
            output_directory=tmp_path.resolve(),
            checkpoint_directory=checkpoint_directory,
            startup_timeout_seconds=120,
            shutdown_grace_seconds=20,
            visible_cuda_models=(),
            logical_cpu_count=configuration.hardware.minimum_logical_cpu_count,
            ram_gib=configuration.hardware.minimum_ram_gib,
            free_disk_gib=configuration.hardware.minimum_free_disk_gib,
            allow_cpu_smoke=True,
        ),
    )
    storage = ReplayShardStorage(
        directory=tmp_path / 'replay',
        maximum_positions_per_shard=configuration.replay.maximum_positions_per_shard,
        capacity_positions=configuration.replay.capacity_positions,
        game_identifier=GameIdentifier.GO,
        payload_schema_version=1,
        compression='none',
        credit_journal=ReplayCreditJournal(tmp_path / 'credits.azc'),
    )
    repository = CheckpointRepository(checkpoint_directory, RUN_ID, CONFIGURATION_SHA256)
    sequence = 0
    trainer: CreditTrainer | None = None

    def publish(records: tuple[ReplayRecord, ...]) -> ShardMetadata:
        nonlocal sequence, trainer
        metadata = storage.publish(sequence, records)
        sequence += 1
        if trainer is None:
            module = create_go_training_module(
                game_configuration=game,
                model_configuration=model,
                objective_configuration=objective_configuration(),
                payload_schema_version=1,
                device=torch.device('cpu'),
                model_initialization_seed=11,
            )
            trainer = CreditTrainer(
                game_module=module,
                replay_storage=storage,
                checkpoint_repository=repository,
                training_configuration=TrainingConfiguration(
                    global_batch_size=1,
                    local_batch_size=1,
                    maximum_optimizer_steps=1,
                    optimizer=AdamWOptimizerConfiguration(
                        kind='adamw',
                        learning_rate=0.001,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-8,
                        weight_decay=0,
                    ),
                    learning_rate_schedule=ConstantLearningRate(kind='constant', multiplier=1),
                    precision='float32',
                    objective=objective_configuration(),
                    checkpoint_every_optimizer_steps=1,
                    gradient_clip_norm=1,
                ),
                credit_configuration=ReplayCreditConfiguration(
                    target_reuse=1,
                    optimizer_steps_per_quantum=1,
                    minimum_positions_before_training=1,
                ),
                root_seed=17,
                rank=TrainingRank(rank=0, world_size=1, device=torch.device('cpu')),
                run_determinism_mode=DeterminismMode.SEEDED_CONCURRENT,
            )
            trainer.train_quantum()
        return metadata

    start_method = 'forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'
    result = RuntimeOrchestrator(
        worker_entrypoint=run_go_worker,
        worker_specifications=tuple(
            specification.model_dump_json().encode() for specification in plan.worker_specifications
        ),
        wall_clock_seconds=plan.duration_seconds,
        startup_timeout_seconds=plan.startup_timeout_seconds,
        shutdown_grace_seconds=plan.shutdown_grace_seconds,
        start_method=start_method,
        replay_publisher=publish,
        topology=plan.topology,
        games_per_shard=plan.games_per_shard,
        telemetry_path=plan.telemetry_path,
        telemetry_write_every_seconds=plan.telemetry_write_every_seconds,
    ).run()

    progress = tuple(message for message in result.messages if isinstance(message, WorkerProgress))
    assert result.orphan_process_ids == ()
    assert result.timed_out
    assert result.elapsed_seconds <= 45
    stopped = next(message for message in result.messages if isinstance(message, WorkerStopped))
    assert storage.credit_journal.credited_unique_positions == stopped.emitted_positions
    assert repository.load_current().manifest.state.replay_credits.model_version == 1
    assert any(isinstance(message, WorkerModelRefreshed) and message.model_version == 1 for message in result.messages)
    assert max(message.interval_maximum_inference_batch_size for message in progress) == 2
    assert plan.telemetry_path.is_file()


def test_long_games_stop_cooperatively_without_emitting_partial_samples(
    tmp_path: Path,
) -> None:
    base = _runtime_configuration()
    configuration = ResolvedRunConfiguration.model_validate(
        base.model_copy(
            update={
                'experiment': base.experiment.model_copy(update={'duration_seconds': 20}),
                'game': base.game.model_copy(update={'safety_ply_cap': 512}),
                'search': base.search.model_copy(
                    update={'budget': FixedSearchBudget(kind='fixed', simulations=100_000)}
                ),
                'replay': base.replay.model_copy(update={'maximum_positions_per_shard': 1_024}),
            }
        ).model_dump()
    )
    checkpoint_directory = (tmp_path / 'checkpoints').resolve()
    plan = build_runtime_plan(
        configuration,
        RuntimeBuildEnvironment(
            run_id=RUN_ID,
            resolved_configuration_sha256=CONFIGURATION_SHA256,
            output_directory=tmp_path.resolve(),
            checkpoint_directory=checkpoint_directory,
            startup_timeout_seconds=120,
            shutdown_grace_seconds=10,
            visible_cuda_models=(),
            logical_cpu_count=configuration.hardware.minimum_logical_cpu_count,
            ram_gib=configuration.hardware.minimum_ram_gib,
            free_disk_gib=configuration.hardware.minimum_free_disk_gib,
            allow_cpu_smoke=True,
        ),
    )
    published: list[tuple[ReplayRecord, ...]] = []

    def publish(records: tuple[ReplayRecord, ...]) -> ShardMetadata:
        published.append(records)
        return ShardMetadata(Path('unused'), len(published) - 1, len(records), 1)

    start_method = 'forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'
    result = RuntimeOrchestrator(
        worker_entrypoint=run_go_worker,
        worker_specifications=tuple(
            specification.model_dump_json().encode() for specification in plan.worker_specifications
        ),
        wall_clock_seconds=plan.duration_seconds,
        startup_timeout_seconds=plan.startup_timeout_seconds,
        shutdown_grace_seconds=plan.shutdown_grace_seconds,
        start_method=start_method,
        replay_publisher=publish,
        topology=plan.topology,
        games_per_shard=plan.games_per_shard,
        telemetry_path=plan.telemetry_path,
        telemetry_write_every_seconds=plan.telemetry_write_every_seconds,
    ).run()

    stopped = next(message for message in result.messages if isinstance(message, WorkerStopped))
    assert result.timed_out
    assert result.elapsed_seconds <= 32
    assert result.orphan_process_ids == ()
    assert stopped.completed_games == 0
    assert stopped.emitted_positions == 0
    assert any(isinstance(message, WorkerProgress) for message in result.messages)
    assert any(isinstance(message, WorkerResourceSample) for message in result.messages)
    assert published == []
