from typing import Generator
import torch
from torch.multiprocessing import Process, Pipe
from pathlib import Path

from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.alpha_zero.train.TrainingStats import TrainingStats
from src.settings import USE_GPU
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.PipeConnection import PipeConnection
from src.util.save_paths import get_latest_model_iteration
from src.cluster.EvaluationProcess import run_evaluation_process
from src.cluster.SelfPlayProcess import run_self_play_process
from src.cluster.TrainerProcess import run_trainer_process

# setup one Trainer Process
# setup one CacheLayer Process
# setup num_inference_nodes InferenceServer Processes
# setup num_self_play_nodes SelfPlay Processes
# setup command process, which notifies InferenceServers to load new models and notifies SelfPlay Processes to write to the next iteration


class CommanderProcess:
    def __init__(self, args: TrainingArgs) -> None:
        self.args = args

        self.trainer_process: Process
        self.commander_trainer_pipe: PipeConnection

        self.self_play_processes: list[Process] = []
        self.commander_self_play_pipes: list[PipeConnection] = []

    def _setup_connections(self) -> None:
        # The Trainer and Commander has a Pipe connection
        # Each SelfPlay and InferenceServer has a Pipe connection to the LoadBalancer
        # The Commander has a Pipe connection to each SelfPlay and InferenceServer

        # Trainer and Commander
        trainer_device_id = 0
        trainer_commander_pipe, self.commander_trainer_pipe = Pipe(duplex=True)

        self.trainer_process = Process(
            target=run_trainer_process, args=(self.args, trainer_commander_pipe, trainer_device_id)
        )
        self.trainer_process.start()

        self.commander_self_play_pipes: list[PipeConnection] = []
        for device_id in range(self.args.cluster.num_self_play_nodes_on_cluster):
            self_play_commander_pipe, commander_self_play_pipe = Pipe(duplex=False)
            self.commander_self_play_pipes.append(commander_self_play_pipe)

            process = Process(
                target=run_self_play_process,
                args=(
                    self.args,
                    self_play_commander_pipe,
                    _get_device_id(device_id, self.args.cluster.num_self_play_nodes_on_cluster),
                ),
            )
            process.start()
            self.self_play_processes.append(process)

    def run(self) -> Generator[tuple[int, TrainingStats], None, None]:
        Path(self.args.save_path).mkdir(parents=True, exist_ok=True)

        log('Setting up connections...')
        self._setup_connections()
        log('Connections set up.')

        starting_iteration = get_latest_model_iteration(self.args.num_iterations, self.args.save_path)
        log(f'Starting training at iteration {starting_iteration}.')

        with log_exceptions('Commander process'):
            for iteration in range(starting_iteration, self.args.num_iterations):
                # send START AT ITERATION: iteration to Trainer and InferenceServers and SelfPlayers
                for pipe in self._all_pipes():
                    pipe.send(f'START AT ITERATION: {iteration}')
                log(f'All processes started at iteration {iteration}.')

                # Wait for Trainer to finish
                train_stats: TrainingStats = self.commander_trainer_pipe.recv()  # type: ignore
                assert self.commander_trainer_pipe.recv() == 'FINISHED'
                yield iteration, train_stats

                # start EvaluationProcess
                Process(
                    target=run_evaluation_process,
                    args=(self.args.evaluation, iteration),
                    daemon=True,
                ).start()

        log('Training complete. Sending STOP to all processes.')
        for pipe in self._all_pipes():
            pipe.send('STOP')

        self.trainer_process.kill()
        for process in self.self_play_processes:
            process.join(timeout=10)

    def _all_processes(self) -> list[Process]:
        return self.self_play_processes + [self.trainer_process]

    def _all_pipes(self) -> list[PipeConnection]:
        return self.commander_self_play_pipes + [self.commander_trainer_pipe]


def _get_device_id(i: int, total: int, num_devices: int = torch.cuda.device_count()) -> int:
    # device 0 should have only half the processes of the other devices as device 0 is 50% occupied by the Trainer
    if not USE_GPU:
        return 0

    num_on_each_device = total / num_devices
    num_on_device_0 = round(num_on_each_device / 2)

    if i < num_on_device_0:
        return 0

    device_id = 1 + (i - num_on_device_0) % (num_devices - 1)
    return device_id
