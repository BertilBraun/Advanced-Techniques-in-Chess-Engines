from typing import Generator
import torch
from torch.multiprocessing import Process, Pipe
from pathlib import Path

from src.train.TrainingArgs import TrainingArgs
from src.train.TrainingStats import TrainingStats
from src.settings import USE_GPU
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.PipeConnection import PipeConnection
from src.util.save_paths import get_latest_model_iteration
from src.cluster.EvaluationProcess import run_evaluation_process
from src.cluster.SelfPlayProcess import run_self_play_process
from src.cluster.TrainerProcess import run_trainer_process


class CommanderProcess:
    """The CommanderProcess is the main process that manages the communication between the Trainer, SelfPlay and InferenceServer processes.

    It starts the Trainer and SelfPlay processes and sends them the current iteration number.
    Once the Trainer is finished, it receives the training stats from the Trainer.
    It then starts the EvaluationProcess and sends the new iteration number to the Trainer and SelfPlay processes.

    Once all iterations are done, it sends a STOP message to all processes and waits for them to finish.
    """

    def __init__(self, run: int, args: TrainingArgs) -> None:
        self.run_id = run
        self.args = args

        self.trainer_process: Process
        self.commander_trainer_pipe: PipeConnection

    def run(self) -> Generator[tuple[int, TrainingStats], None, None]:
        """The main loop of the CommanderProcess. The resulting generator yields after each iteration. If the Generator is not consumed, no further iterations will be trained."""

        Path(self.args.save_path).mkdir(parents=True, exist_ok=True)

        log('Setting up connections...')
        # Trainer and Commander
        trainer_device_id = torch.cuda.device_count() - 1
        trainer_commander_pipe, self.commander_trainer_pipe = Pipe(duplex=True)

        self.trainer_process = Process(
            target=run_trainer_process, args=(self.run_id, self.args, trainer_commander_pipe, trainer_device_id)
        )
        self.trainer_process.start()
        log('Connections set up.')

        starting_iteration = get_latest_model_iteration(self.args.save_path)
        log(f'Starting training at iteration {starting_iteration}.')

        with log_exceptions('Commander process'):
            for iteration in range(starting_iteration, self.args.num_iterations):
                # send START AT ITERATION: iteration to Trainer and InferenceServers and SelfPlayers
                self.commander_trainer_pipe.send(f'START AT ITERATION: {iteration}')

                # Wait for Trainer to finish
                train_stats: TrainingStats = self.commander_trainer_pipe.recv()  # type: ignore
                assert self.commander_trainer_pipe.recv() == 'FINISHED'
                yield iteration, train_stats

                # start EvaluationProcess
                Process(target=run_evaluation_process, args=(self.run_id, self.args, iteration), daemon=True).start()

        log('Training complete. Sending STOP to all processes.')
        self.commander_trainer_pipe.send('STOP')

        self.trainer_process.kill()
        exit()
