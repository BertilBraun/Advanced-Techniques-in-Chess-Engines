import random
import time
from pathlib import Path
import numpy as np
import torch

from src.cluster.TrainerProcess import number_of_games_in_iteration
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TensorboardWriter, USE_GPU, USE_CPP, TRAINING_ARGS
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
    refresh_self_play_model_message,
    self_play_model_refreshed_message,
    update_self_play_search_schedule_message,
)
from src.util.log import log
from src.util.exceptions import log_exceptions
from src.train.TrainingArgs import TrainingArgs
from src.train.CreditPublication import PublicationValidationScope, load_credit_publication_pointer
from src.train.RollingReplayBuffer import commit_replay_shard
from src.util.profiler import start_cpu_usage_logger
from src.util.background_worker import BackgroundWorker
from src.util.save_paths import model_save_path
from src.util.timing import reset_times

if USE_CPP:
    from src.self_play.SelfPlayCpp import SelfPlayCpp as SelfPlay
else:
    from src.self_play.SelfPlayPy import SelfPlayPy as SelfPlay


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
        current_iteration = -1
        running = False
        paused = False
        credit_training = self.args.training.credit_training
        usage_logger: BackgroundWorker | None = None

        try:
            while True:
                if running:
                    with log_exceptions('Self playing failed'):
                        self.self_play.self_play()

                    if self.self_play.dataset.stats.num_games >= self.args.self_play.num_games_after_which_to_write:
                        self.flush_replay_shard(current_iteration)

                    if credit_training is None and (
                        number_of_games_in_iteration(current_iteration, self.args.save_path)
                        >= TRAINING_ARGS.num_games_per_iteration * 5
                    ):
                        log(f'Iteration {current_iteration} has enough games.')
                        self.flush_replay_shard(current_iteration)
                        running = False
                else:
                    time.sleep(0.1)  # Sleep to avoid busy waiting

                if self.communication.is_received('STOP'):
                    break
                if self.communication.try_receive_from_id('START USAGE LOGGER', self.node_id):
                    if usage_logger is not None:
                        usage_logger.stop()
                    usage_logger = start_cpu_usage_logger(self.run_id, 'self_play')
                if self.communication.try_receive_from_id(STOP_SELF_PLAY, self.node_id):
                    self.flush_replay_shard(current_iteration)
                    running = False
                    paused = True
                    self.communication.send_to_id(SELF_PLAY_PAUSED, self.node_id)
                if self.communication.try_receive_from_id(RESUME_SELF_PLAY, self.node_id):
                    paused = False
                    running = current_iteration >= 0
                    self.communication.send_to_id(SELF_PLAY_RESUMED, self.node_id)
                continuous_running = self._continuous_self_play_state(current_iteration)
                if continuous_running is not None:
                    paused = False
                    running = continuous_running
                if self.communication.try_receive_from_id(FLUSH_REPLAY_SHARD, self.node_id):
                    self.flush_replay_shard(current_iteration)
                if self.communication.try_receive_from_id(
                    SNAPSHOT_SELF_PLAY_STATISTICS,
                    self.node_id,
                ):
                    self.self_play.snapshot_statistics(current_iteration)

                self._update_search_schedule_if_requested()
                current_iteration = self._refresh_model_if_requested(current_iteration)

                legacy_iterations = (
                    range(self.args.num_iterations, current_iteration, -1)
                    if credit_training is None and not paused
                    else ()
                )
                for iteration in legacy_iterations:
                    start_recieved = self.communication.is_received(f'START AT ITERATION: {iteration}')
                    load_received = self.communication.is_received(f'LOAD MODEL: {iteration}')
                    if start_recieved:
                        self.flush_replay_shard(current_iteration)
                        current_iteration = iteration
                        running = True
                        reset_times()
                    if load_received:
                        self.flush_replay_shard(current_iteration)
                        if current_iteration >= 0:
                            self.self_play.snapshot_statistics(current_iteration)
                        self.self_play.update_search_schedule(self.self_play.search_schedule(iteration))
                        self.self_play.refresh_model(
                            iteration,
                            model_save_path(iteration, self.args.save_path).with_suffix('.jit.pt'),
                        )
                        current_iteration = iteration

                    if start_recieved or load_received:
                        break

                self.communication.send_heartbeat(f'SELF PLAY {self.node_id}')
        finally:
            if usage_logger is not None:
                usage_logger.stop()

        log('Self play process stopped.')

    def _refresh_model_if_requested(self, current_model_version: int) -> int:
        if self.args.training.credit_training is not None:
            serialized_pointer = self.communication.try_read(LATEST_SELF_PLAY_MODEL_VERSION)
            if serialized_pointer is None:
                return current_model_version
            if serialized_pointer == self.loaded_credit_publication_pointer:
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
            self.loaded_credit_jit_sha256 = publication.jit_model.sha256
            self.loaded_credit_publication_pointer = serialized_pointer
            self.communication.send_value_to_id(
                self_play_model_refreshed_message(model_version),
                self.node_id,
                publication.jit_model.sha256,
            )
            return model_version

        for model_version in range(self.args.num_iterations, current_model_version, -1):
            if not self.communication.is_received(refresh_self_play_model_message(model_version)):
                continue
            self.self_play.refresh_model(
                model_version,
                model_save_path(model_version, self.args.save_path).with_suffix('.jit.pt'),
            )
            self.communication.send_to_id(
                self_play_model_refreshed_message(model_version),
                self.node_id,
            )
            return model_version
        return current_model_version

    def _maximum_model_version(self) -> int:
        credit_training = self.args.training.credit_training
        if credit_training is None:
            return self.args.num_iterations
        return credit_training.maximum_optimizer_steps // credit_training.optimizer_steps_per_quantum

    def _continuous_self_play_state(self, current_model_version: int) -> bool | None:
        if not self.communication.is_received(START_CONTINUOUS_SELF_PLAY):
            return None
        if self.args.training.credit_training is None:
            raise RuntimeError('Continuous self-play requires credit-driven training.')
        return current_model_version >= 0

    def _update_search_schedule_if_requested(self) -> None:
        current_schedule = self.self_play.search_schedule_state
        current_schedule_version = current_schedule.schedule_version if current_schedule is not None else -1
        if self.args.training.credit_training is not None:
            return
        for schedule_version in range(
            self.args.num_iterations,
            current_schedule_version,
            -1,
        ):
            if not self.communication.is_received(update_self_play_search_schedule_message(schedule_version)):
                continue
            self.self_play.update_search_schedule(self.self_play.search_schedule(schedule_version))
            return

    def flush_replay_shard(self, iteration: int) -> None:
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
            default=iteration,
        )
        maximum_model_version = max(
            (maximum for _, maximum in model_version_ranges),
            default=iteration,
        )
        commit_replay_shard(
            dataset=self.self_play.dataset,
            replay_inbox=Path(self.args.save_path) / 'replay_inbox',
            producing_worker=self.node_id,
            minimum_model_version=minimum_model_version,
            maximum_model_version=maximum_model_version,
        )
        self.last_flushed_completed_searches = completed_searches
        self.self_play.dataset = SelfPlayDataset()
