import random
import time
from pathlib import Path
import numpy as np
import torch

from src.settings import TensorboardWriter, USE_GPU
from src.util.communication import (
    Communication,
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
from src.util.profiler import start_cpu_usage_logger
from src.util.background_worker import BackgroundWorker
from src.self_play.SelfPlay import SelfPlay
from src.self_play.chess_completed_game import ChessCompletedGamePublisher


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
            enabled=node_id < args.topology.self_play.tensorboard_processes,
        ),
    ):
        self_play_process.run()


class SelfPlayProcess:
    """This class provides functionality to run the self play process. It runs self play games and saves the dataset to disk. It listens to the commander for messages to start and stop the self play process."""

    def __init__(
        self, args: TrainingArgs, communication_folder: str, device_id: int, node_id: int, run_id: int
    ) -> None:
        self.args = args
        self.completed_game_publisher = ChessCompletedGamePublisher(
            Path(args.save_path),
            run_id,
            node_id,
        )
        self.self_play = SelfPlay(device_id, args, self.completed_game_publisher)
        self.communication = Communication(communication_folder)
        self.node_id = node_id
        self.run_id = run_id
        self.loaded_credit_jit_sha256: str | None = None
        self.loaded_credit_publication_pointer: str | None = None

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

                else:
                    time.sleep(0.1)  # Sleep to avoid busy waiting

                if self.communication.is_received('STOP'):
                    break
                if self.communication.try_receive_from_id('START USAGE LOGGER', self.node_id):
                    if usage_logger is not None:
                        usage_logger.stop()
                    usage_logger = start_cpu_usage_logger(self.run_id, 'self_play')
                if self.communication.try_receive_from_id(STOP_SELF_PLAY, self.node_id):
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
        self.loaded_credit_jit_sha256 = publication.jit_model.sha256
        self.loaded_credit_publication_pointer = serialized_pointer
        self.communication.send_value_to_id(
            self_play_model_refreshed_message(model_version),
            self.node_id,
            publication.jit_model.sha256,
        )
        return model_version

    def _maximum_model_version(self) -> int:
        parameters = self.args.lifecycle.credit
        return parameters.maximum_optimizer_steps // parameters.optimizer_steps_per_quantum

    def _continuous_self_play_state(self, current_model_version: int) -> bool | None:
        if not self.communication.is_received(START_CONTINUOUS_SELF_PLAY):
            return None
        return current_model_version >= 0
