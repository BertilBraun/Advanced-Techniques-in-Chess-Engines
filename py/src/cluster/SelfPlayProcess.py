import random
import time
import json
from math import ceil
from pathlib import Path
import h5py
import numpy as np
import torch

from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TensorboardWriter, USE_GPU, log_scalar
from src.util.communication import (
    Communication,
    FLUSH_REPLAY_SHARD,
    LATEST_SELF_PLAY_MODEL_VERSION,
    RESUME_SELF_PLAY,
    SELF_PLAY_PAUSED,
    SELF_PLAY_RESUMED,
    SNAPSHOT_SELF_PLAY_STATISTICS,
    START_CONTINUOUS_SELF_PLAY,
    STOP_SELF_PLAY,
    self_play_model_refreshed_message,
)
from src.util.log import log
from src.util.exceptions import log_exceptions
from src.train.TrainingArgs import TrainingArgs
from src.train.CreditPublication import PublicationValidationScope, load_credit_publication_pointer
from src.train.RollingReplayBuffer import commit_replay_shard
from src.train.ReplayReanalysis import ReanalysisPosition, write_reanalysis_sidecar
from src.util.profiler import start_cpu_usage_logger
from src.util.background_worker import BackgroundWorker
from src.self_play.SelfPlay import SelfPlay


def run_self_play_process(
    run: int, args: TrainingArgs, communication_folder: str, device_id: int, node_id: int
) -> None:
    if USE_GPU:
        # torch.cuda.set_per_process_memory_fraction(1 / 64, device=device_id)
        torch.cuda.set_device(device_id)

    worker_seed = args.random_seed + node_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)

    self_play_process = SelfPlayProcess(args, communication_folder, device_id=device_id, node_id=node_id, run_id=run)
    with (
        log_exceptions(f'Self play process {node_id} crashed.'),
        TensorboardWriter(
            run,
            'self_play',
            postfix_pid=False,
            enabled=node_id < args.cluster.self_play_tensorboard_processes,
        ),
    ):
        self_play_process.run()


class SelfPlayProcess:
    """This class provides functionality to run the self play process. It runs self play games and saves the dataset to disk. It listens to the commander for messages to start and stop the self play process."""

    def __init__(
        self, args: TrainingArgs, communication_folder: str, device_id: int, node_id: int, run_id: int
    ) -> None:
        self.args = args
        self.self_play = SelfPlay(device_id, args)
        self.communication = Communication(communication_folder)
        self.node_id = node_id
        self.run_id = run_id
        self.loaded_credit_jit_sha256: str | None = None
        self.loaded_credit_publication_pointer: str | None = None
        self.last_flushed_completed_searches = 0

    def run(self) -> None:
        current_model_version = -1
        running = False
        paused = False
        usage_logger: BackgroundWorker | None = None

        try:
            while True:
                if running:
                    with log_exceptions('Self playing failed'):
                        self.self_play.self_play()

                    if self.self_play.dataset.stats.num_games >= self.args.self_play.num_games_after_which_to_write:
                        self.flush_replay_shard(current_model_version)
                else:
                    time.sleep(0.1)  # Sleep to avoid busy waiting

                if self.communication.is_received('STOP'):
                    break
                if self.communication.try_receive_from_id('START USAGE LOGGER', self.node_id):
                    if usage_logger is not None:
                        usage_logger.stop()
                    usage_logger = start_cpu_usage_logger(self.run_id, 'self_play')
                if self.communication.try_receive_from_id(STOP_SELF_PLAY, self.node_id):
                    self.flush_replay_shard(current_model_version)
                    running = False
                    paused = True
                    self.communication.send_to_id(SELF_PLAY_PAUSED, self.node_id)
                if self.communication.try_receive_from_id(RESUME_SELF_PLAY, self.node_id):
                    paused = False
                    running = current_model_version >= 0
                    self.communication.send_to_id(SELF_PLAY_RESUMED, self.node_id)
                if not paused:
                    continuous_running = self._continuous_self_play_state(current_model_version)
                    if continuous_running is not None:
                        running = continuous_running
                if self.communication.try_receive_from_id(FLUSH_REPLAY_SHARD, self.node_id):
                    self.flush_replay_shard(current_model_version)
                if self.communication.try_receive_from_id(
                    SNAPSHOT_SELF_PLAY_STATISTICS,
                    self.node_id,
                ):
                    self.self_play.snapshot_statistics(current_model_version)

                current_model_version = self._refresh_model_if_requested(current_model_version)

                self.communication.send_heartbeat(f'SELF PLAY {self.node_id}')
        finally:
            if usage_logger is not None:
                usage_logger.stop()

        log('Self play process stopped.')

    def _refresh_model_if_requested(self, current_model_version: int) -> int:
        serialized_pointer = self.communication.try_read(LATEST_SELF_PLAY_MODEL_VERSION)
        if serialized_pointer is None or serialized_pointer == self.loaded_credit_publication_pointer:
            return current_model_version
        _, publication = load_credit_publication_pointer(
            Path(self.args.save_path),
            serialized_pointer,
            PublicationValidationScope.JIT_ONLY,
        )
        model_version = publication.model_version
        if model_version > self._maximum_model_version():
            raise ValueError(f'Published model version exceeds the configured maximum: {model_version}')
        if model_version < current_model_version:
            raise ValueError('Published model version cannot move backwards.')
        if model_version == current_model_version:
            if self.loaded_credit_jit_sha256 != publication.jit_model.sha256:
                raise ValueError('Published model version changed immutable JIT identity.')
            return current_model_version
        self.self_play.update_search_schedule(self.self_play.search_schedule(model_version))
        self.self_play.refresh_model(
            model_version,
            Path(self.args.save_path) / publication.jit_model.path,
        )
        if self.node_id == 0:
            self._reanalyse_recent_replay(model_version)
        self.loaded_credit_jit_sha256 = publication.jit_model.sha256
        self.loaded_credit_publication_pointer = serialized_pointer
        self.communication.send_value_to_id(
            self_play_model_refreshed_message(model_version),
            self.node_id,
            publication.jit_model.sha256,
        )
        return model_version

    def _reanalyse_recent_replay(self, model_version: int) -> None:
        fraction = self.args.self_play.replay_reanalysis_fraction
        if fraction <= 0.0:
            return
        replay_inbox = Path(self.args.save_path) / 'replay_inbox'
        payloads = sorted(
            (path for path in replay_inbox.glob('*.hdf5') if '.reanalysis-' not in path.name),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for payload_path in payloads:
            sidecar = payload_path.with_name(f'{payload_path.stem}.reanalysis-{model_version:010d}.hdf5')
            if sidecar.exists():
                continue
            with h5py.File(payload_path, 'r') as file:
                starting_fens = np.asarray(file['position_starting_fens'].asstr()[...])
                moves_json = np.asarray(file['position_moves_uci'].asstr()[...])
            eligible_indices = np.flatnonzero(starting_fens != '')
            if not len(eligible_indices):
                continue
            requested_count = min(
                self.args.self_play.replay_reanalysis_maximum_positions_per_refresh,
                max(1, ceil(len(eligible_indices) * fraction)),
            )
            generator = np.random.default_rng((model_version << 16) + self.node_id)
            selected_indices = generator.choice(
                eligible_indices,
                size=requested_count,
                replace=False,
            )
            positions = tuple(
                ReanalysisPosition(
                    row_index=int(row_index),
                    starting_fen=str(starting_fens[row_index]),
                    moves_uci=tuple(json.loads(str(moves_json[row_index]))),
                )
                for row_index in selected_indices
            )
            started = time.perf_counter()
            targets = self.self_play.reanalyse_positions(positions)
            write_reanalysis_sidecar(payload_path, model_version, targets)
            duration = time.perf_counter() - started
            log_scalar('reanalysis/positions_refreshed', len(targets), model_version)
            log_scalar('reanalysis/duration_seconds', duration, model_version)
            log_scalar(
                'reanalysis/positions_per_second',
                len(targets) / duration if duration else 0.0,
                model_version,
            )
            log_scalar('reanalysis/source_rows', len(starting_fens), model_version)
            return

    def _maximum_model_version(self) -> int:
        parameters = self.args.training.credit_training
        return parameters.maximum_optimizer_steps // parameters.optimizer_steps_per_quantum

    def _continuous_self_play_state(self, current_model_version: int) -> bool | None:
        if not self.communication.is_received(START_CONTINUOUS_SELF_PLAY):
            return None
        return current_model_version >= 0

    def flush_replay_shard(self, model_version: int) -> None:
        if not len(self.self_play.dataset):
            return

        completed_searches = self.self_play.completed_searches
        shard_completed_searches = completed_searches - self.last_flushed_completed_searches
        if shard_completed_searches < 0:
            raise RuntimeError('Self-play completed-search count cannot move backwards.')
        self.self_play.dataset.stats = self.self_play.dataset.stats.overwrite(
            completed_searches=shard_completed_searches
        )
        model_version_ranges = self.self_play.dataset.stats.game_model_version_ranges
        minimum_model_version = min(
            (minimum for minimum, _ in model_version_ranges),
            default=model_version,
        )
        maximum_model_version = max(
            (maximum for _, maximum in model_version_ranges),
            default=model_version,
        )
        manifest = commit_replay_shard(
            dataset=self.self_play.dataset,
            replay_inbox=Path(self.args.save_path) / 'replay_inbox',
            producing_worker=self.node_id,
            minimum_model_version=minimum_model_version,
            maximum_model_version=maximum_model_version,
        )
        log_scalar('replay/raw_rows_per_shard', manifest.raw_sample_count, model_version)
        log_scalar('replay/unique_groups_per_shard', manifest.unique_sample_count, model_version)
        log_scalar('replay/duplicate_factor_per_shard', manifest.duplicate_factor, model_version)
        log_scalar(
            'replay/effective_multiplicity_weight_per_shard',
            manifest.effective_multiplicity_weight,
            model_version,
        )
        log_scalar(
            'replay/conflicting_target_groups_per_shard',
            manifest.conflicting_target_groups,
            model_version,
        )
        self.last_flushed_completed_searches = completed_searches
        self.self_play.dataset = SelfPlayDataset()
