from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import psutil
import torch

from src.Encoding import BINARY_CHANNELS, C, H, SCALAR_CHANNELS, W, encode_board_state
from src.cluster.TrainerProcess import TrainerProcess
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.settings import CurrentGame, TRAINING_ARGS
from src.train.TrainingArgs import ClusterParams, TrainingArgs
from src.util.save_paths import create_model, create_optimizer, save_model_and_optimizer


@dataclass(frozen=True)
class Arguments:
    device_ids: tuple[int, ...]
    samples: int
    replay_source_directory: Path | None
    work_directory: Path
    output_path: Path
    monitor_interval_seconds: float


@dataclass(frozen=True)
class GpuUtilization:
    device_id: int
    samples: int
    mean_sm_percent: float
    mean_memory_controller_percent: float
    peak_memory_mib: float


@dataclass(frozen=True)
class SmokeResult:
    source_revision: str
    replay_source: str
    device_ids: tuple[int, ...]
    global_batch_size: int
    local_batch_size: int
    replay_samples: int
    retained_samples: int
    dropped_samples: int
    optimizer_steps: int
    aggregated_training_samples: int
    training_phase_seconds: float
    optimizer_seconds: float
    production_phase_samples_per_second: float
    optimizer_samples_per_second: float
    peak_host_rss_mib: float
    gpu_utilization: tuple[GpuUtilization, ...]


class ResourceMonitor:
    def __init__(self, device_ids: tuple[int, ...], interval_seconds: float) -> None:
        self.device_ids = device_ids
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='ddp-resource-monitor')
        self.host_rss_samples: list[float] = []
        self.gpu_samples: dict[int, list[tuple[float, float, float]]] = {device_id: [] for device_id in device_ids}
        self.error: BaseException | None = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Resource monitor did not stop.')
        if self.error is not None:
            raise RuntimeError('Resource monitor failed.') from self.error

    def result(self) -> tuple[float, tuple[GpuUtilization, ...]]:
        peak_host_rss_mib = max(self.host_rss_samples, default=0.0)
        utilization = tuple(
            GpuUtilization(
                device_id=device_id,
                samples=len(self.gpu_samples[device_id]),
                mean_sm_percent=float(np.mean([sample[0] for sample in self.gpu_samples[device_id]])),
                mean_memory_controller_percent=float(np.mean([sample[1] for sample in self.gpu_samples[device_id]])),
                peak_memory_mib=max(sample[2] for sample in self.gpu_samples[device_id]),
            )
            for device_id in self.device_ids
            if self.gpu_samples[device_id]
        )
        return peak_host_rss_mib, utilization

    def _run(self) -> None:
        try:
            root_process = psutil.Process()
            while not self.stop_event.is_set():
                processes = (root_process, *root_process.children(recursive=True))
                rss_bytes = sum(resident_memory_bytes(process) for process in processes)
                self.host_rss_samples.append(rss_bytes / 2**20)
                completed = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=index,utilization.gpu,utilization.memory,memory.used',
                        '--format=csv,noheader,nounits',
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for line in completed.stdout.splitlines():
                    device_id_text, sm_text, memory_controller_text, memory_used_text = (
                        field.strip() for field in line.split(',')
                    )
                    device_id = int(device_id_text)
                    if device_id in self.gpu_samples:
                        self.gpu_samples[device_id].append(
                            (float(sm_text), float(memory_controller_text), float(memory_used_text))
                        )
                self.stop_event.wait(self.interval_seconds)
        except BaseException as error:
            self.error = error


def resident_memory_bytes(process: psutil.Process) -> int:
    try:
        return process.memory_info().rss
    except psutil.NoSuchProcess:
        return 0


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-ids', nargs='+', required=True, type=int)
    parser.add_argument('--samples', default=409_600, type=int)
    parser.add_argument('--replay-source-directory', type=Path)
    parser.add_argument('--work-directory', required=True, type=Path)
    parser.add_argument('--output-path', required=True, type=Path)
    parser.add_argument('--monitor-interval-seconds', default=0.5, type=float)
    namespace = parser.parse_args()
    device_ids = tuple(namespace.device_ids)
    if not device_ids or len(set(device_ids)) != len(device_ids):
        raise ValueError('Device IDs must be nonempty and unique.')
    if 2048 % len(device_ids) != 0:
        raise ValueError('The 2,048-sample global batch must divide evenly across trainer devices.')
    if namespace.samples <= 0:
        raise ValueError('Sample count must be positive.')
    if namespace.replay_source_directory is not None and not namespace.replay_source_directory.is_dir():
        raise ValueError(f'Replay source directory does not exist: {namespace.replay_source_directory}')
    if namespace.monitor_interval_seconds <= 0:
        raise ValueError('Monitor interval must be positive.')
    return Arguments(
        device_ids=device_ids,
        samples=namespace.samples,
        replay_source_directory=namespace.replay_source_directory,
        work_directory=namespace.work_directory,
        output_path=namespace.output_path,
        monitor_interval_seconds=namespace.monitor_interval_seconds,
    )


def source_revision() -> str:
    completed = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def encoded_state_templates(count: int, seed: int) -> tuple[bytes, ...]:
    generator = np.random.default_rng(seed)
    templates: list[bytes] = []
    for _ in range(count):
        state = np.zeros((C, H, W), dtype=np.int8)
        state[list(BINARY_CHANNELS)] = generator.integers(
            0,
            2,
            size=(len(BINARY_CHANNELS), H, W),
            dtype=np.int8,
        )
        state[list(SCALAR_CHANNELS)] = generator.integers(
            -1,
            2,
            size=(len(SCALAR_CHANNELS), 1, 1),
            dtype=np.int8,
        )
        templates.append(encode_board_state(state))
    return tuple(templates)


def write_replay_fixture(path: Path, sample_count: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    templates = encoded_state_templates(min(sample_count, 4096), seed)
    states = np.asarray([templates[index % len(templates)] for index in range(sample_count)])
    visit_counts = np.zeros((sample_count, 1, 2), dtype=np.int32)
    visit_counts[:, 0, 0] = np.arange(sample_count) % CurrentGame.action_size
    visit_counts[:, 0, 1] = 600
    value_targets = np.linspace(-1.0, 1.0, num=sample_count, dtype=np.float32)
    game_count = max(1, sample_count // 50)
    stats = SelfPlayDatasetStats(
        num_samples=sample_count,
        num_games=game_count,
        game_lengths=[50] * game_count,
        total_generation_time=float(game_count),
    )
    with h5py.File(path, 'w') as file:
        file.create_dataset('states', data=states)
        file.create_dataset('visit_counts', data=visit_counts)
        file.create_dataset('value_targets', data=value_targets)
        file.attrs['metadata'] = str(SelfPlayDataset._get_current_metadata())
        file.attrs['stats'] = str(stats._asdict())


def stage_replay_files(source_directory: Path, destination_directory: Path, minimum_samples: int) -> int:
    source_files = sorted(source_directory.glob('*.hdf5'))
    if not source_files:
        raise ValueError(f'Replay source directory contains no HDF5 files: {source_directory}')

    destination_directory.mkdir(parents=True, exist_ok=False)
    sample_count = 0
    for source_file in source_files:
        with h5py.File(source_file, 'r') as file:
            file_samples = int(file['states'].shape[0])
        shutil.copy2(source_file, destination_directory / source_file.name)
        sample_count += file_samples
        if sample_count >= minimum_samples:
            return sample_count
    raise ValueError(f'Replay source contains {sample_count:,} samples, fewer than the requested {minimum_samples:,}.')


def smoke_arguments(arguments: Arguments) -> TrainingArgs:
    training_args = copy.deepcopy(TRAINING_ARGS)
    training_args.save_path = str(arguments.work_directory)
    training_args.num_iterations = 1
    training_args.num_games_per_iteration = max(1, arguments.samples // 50)
    training_args.training.num_epochs = 1
    training_args.training.global_batch_size = 2048
    training_args.training.local_batch_size = 2048 // len(arguments.device_ids)
    training_args.training.num_workers = 0
    training_args.cluster = ClusterParams(
        trainer_device_type='cuda',
        trainer_process_group_backend='nccl',
        trainer_rank_zero_device_id=arguments.device_ids[0],
        trainer_ddp_device_ids=arguments.device_ids,
        evaluation_device_cycle=(arguments.device_ids[0],),
        self_play_device_ids=(arguments.device_ids[0],),
        self_play_tensorboard_processes=1,
        trainer_cpu_threads=8,
        trainer_interop_threads=2,
        self_play_node_ids_to_pause_during_training=(),
        max_concurrent_evaluations=1,
    )
    return training_args


def prepare_smoke_run(arguments: Arguments, training_args: TrainingArgs) -> int:
    if arguments.work_directory.exists():
        raise ValueError(f'Smoke work directory already exists: {arguments.work_directory}')
    arguments.work_directory.mkdir(parents=True)
    os.environ['TRAINING_TENSORBOARD_LOG_PATH'] = str(arguments.work_directory / 'logs')
    os.environ['TRAINING_TENSORBOARD_RUN_DIRECTORY'] = 'production-ddp-smoke'
    replay_directory = arguments.work_directory / 'memory_0'
    if arguments.replay_source_directory is None:
        write_replay_fixture(replay_directory / 'smoke.hdf5', arguments.samples, training_args.random_seed)
        replay_samples = arguments.samples
    else:
        replay_samples = stage_replay_files(
            arguments.replay_source_directory,
            replay_directory,
            arguments.samples,
        )
    device = torch.device('cuda', arguments.device_ids[0])
    torch.cuda.set_device(device)
    model = create_model(training_args.network, device)
    optimizer = create_optimizer(model, training_args.training.optimizer)
    save_model_and_optimizer(model, optimizer, 0, training_args.save_path)
    return replay_samples


def run_smoke(arguments: Arguments) -> SmokeResult:
    training_args = smoke_arguments(arguments)
    replay_samples = prepare_smoke_run(arguments, training_args)
    trainer = TrainerProcess(training_args, run_id=9999, starting_iteration=0)
    try:
        trainer.load_all_memories_to_train_on_for_iteration(0)
        monitor = ResourceMonitor(arguments.device_ids, arguments.monitor_interval_seconds)
        monitor.start()
        try:
            started_at = time.perf_counter()
            training_stats = trainer.train(0)
            training_phase_seconds = time.perf_counter() - started_at
        finally:
            monitor.stop()
        peak_host_rss_mib, gpu_utilization = monitor.result()
    finally:
        trainer.close()

    retained_samples = replay_samples - replay_samples % training_args.training.global_batch_size
    result = SmokeResult(
        source_revision=source_revision(),
        replay_source=(
            str(arguments.replay_source_directory.resolve())
            if arguments.replay_source_directory is not None
            else 'generated synthetic fixture'
        ),
        device_ids=arguments.device_ids,
        global_batch_size=training_args.training.global_batch_size,
        local_batch_size=training_args.training.local_batch_size,
        replay_samples=replay_samples,
        retained_samples=retained_samples,
        dropped_samples=replay_samples - retained_samples,
        optimizer_steps=retained_samples // training_args.training.global_batch_size,
        aggregated_training_samples=training_stats.sample_count,
        training_phase_seconds=training_phase_seconds,
        optimizer_seconds=trainer.last_optimizer_seconds,
        production_phase_samples_per_second=training_stats.sample_count / training_phase_seconds,
        optimizer_samples_per_second=training_stats.sample_count / trainer.last_optimizer_seconds,
        peak_host_rss_mib=peak_host_rss_mib,
        gpu_utilization=gpu_utilization,
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_path.write_text(json.dumps(asdict(result), indent=2) + '\n', encoding='utf-8')
    return result


def main() -> None:
    print(json.dumps(asdict(run_smoke(parse_arguments())), indent=2))


if __name__ == '__main__':
    main()
