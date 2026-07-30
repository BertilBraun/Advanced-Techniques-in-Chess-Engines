from __future__ import annotations

from pathlib import PurePosixPath

from src.az.config.base import DeterminismMode
from src.az.config.experiment import HardwareConfiguration, ManifestPolicy
from src.az.config.model import FixedModelSchedule, ModelConfiguration
from src.az.config.resolution import (
    AuthoringExperimentConfiguration,
    AuthoringRunConfiguration,
    AuthoringSearchConfiguration,
    resolve_configuration,
)
from src.az.config.root import ResolvedRunConfiguration
from src.az.config.runtime import DeviceAssignment, TopologyConfiguration
from src.az.config.search import (
    DisabledRootExploration,
    FixedSearchBudget,
    SearchInferenceConfiguration,
)
from src.az.config.training import ReplayCreditConfiguration
from src.az.games.go.configuration import ResidualGoModelConfiguration


def local_cpu_smoke_configuration() -> ResolvedRunConfiguration:
    authoring = AuthoringRunConfiguration(
        experiment=AuthoringExperimentConfiguration(
            name='go-local-readiness',
            arm_id='local-cpu-smoke',
            hypothesis='The real Go pipeline completes a bounded CPU lifecycle.',
            root_seed=2026073002,
            duration_seconds=12,
            checkpoint_elapsed_seconds=(12,),
            output_directory=PurePosixPath('runs/go-local-readiness'),
            manifest_policy=ManifestPolicy(
                require_clean_source=False,
                record_dependency_versions=True,
                determinism_mode=DeterminismMode.SEEDED_CONCURRENT,
            ),
        ),
        search=AuthoringSearchConfiguration(
            budget=FixedSearchBudget(kind='fixed', simulations=1),
            root_exploration=DisabledRootExploration(kind='disabled'),
            inference=SearchInferenceConfiguration(
                maximum_batch_size=1,
                maximum_wait_microseconds=1_000,
                cache_capacity=32,
            ),
        ),
        hardware=HardwareConfiguration(
            profile_name='local-cpu-smoke',
            provider='local',
            offer_id='cpu-readiness',
            expected_gpu_model='none',
            expected_gpu_count=0,
            minimum_logical_cpu_count=1,
            minimum_ram_gib=1,
            minimum_free_disk_gib=1,
            hourly_cost=0,
            currency='EUR',
        ),
        topology=TopologyConfiguration(
            trainer=DeviceAssignment(device_ids=(0,)),
            self_play=DeviceAssignment(device_ids=(0,)),
            evaluation=DeviceAssignment(device_ids=(0,)),
            self_play_workers_per_device=1,
            maximum_active_searches_per_worker=1,
            inference_workers_per_device=1,
            inference_batch_size=1,
            maximum_pending_inference_batches=1,
            data_loader_workers_per_rank=0,
            evaluation_concurrency=1,
        ),
    )
    tiny_architecture = ResidualGoModelConfiguration(
        family='residual_go',
        channels=4,
        residual_blocks=1,
        policy_channels=2,
        value_hidden_size=8,
        normalization='batch',
        activation='relu',
    )
    authoring = authoring.model_copy(
        update={
            'game': authoring.game.model_copy(
                update={
                    'board_size': 3,
                    'safety_ply_cap': 9,
                    'history_length': 2,
                }
            ),
            'model': ModelConfiguration(schedule=FixedModelSchedule(kind='fixed', architecture=tiny_architecture)),
            'self_play': authoring.self_play.model_copy(update={'games_per_shard': 1}),
            'replay': authoring.replay.model_copy(
                update={
                    'capacity_positions': 90,
                    'maximum_positions_per_shard': 9,
                    'credits': ReplayCreditConfiguration(
                        target_reuse=1,
                        optimizer_steps_per_quantum=1,
                        minimum_positions_before_training=1,
                    ),
                }
            ),
            'training': authoring.training.model_copy(
                update={
                    'global_batch_size': 1,
                    'local_batch_size': 1,
                    'maximum_optimizer_steps': 1,
                    'precision': 'float32',
                    'checkpoint_every_optimizer_steps': 1,
                }
            ),
            'evaluation': authoring.evaluation.model_copy(
                update={
                    'search': authoring.evaluation.search.model_copy(
                        update={
                            'budget': FixedSearchBudget(kind='fixed', simulations=1),
                            'inference': SearchInferenceConfiguration(
                                maximum_batch_size=1,
                                maximum_wait_microseconds=1_000,
                                cache_capacity=32,
                            ),
                        }
                    ),
                    'paired_games_per_checkpoint': 2,
                    'bootstrap_samples': 10,
                }
            ),
            'telemetry': authoring.telemetry.model_copy(
                update={'write_every_seconds': 1, 'resource_sample_every_seconds': 1}
            ),
        }
    )
    return resolve_configuration(AuthoringRunConfiguration.model_validate(authoring.model_dump()))
