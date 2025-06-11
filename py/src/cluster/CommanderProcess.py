from typing import Generator
import torch
from torch.multiprocessing import Process, Pipe
from pathlib import Path

from src.eval.ModelEvaluationCpp import ModelEvaluation
from src.train.TrainingArgs import TrainingArgs
from src.train.TrainingStats import TrainingStats
from src.settings import USE_GPU
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.PipeConnection import PipeConnection
from src.util.save_paths import (
    get_latest_model_iteration,
    load_model_and_optimizer,
    model_save_path,
    save_model_and_optimizer,
)
from src.cluster.EvaluationProcess import run_evaluation_process
from src.cluster.SelfPlayProcess import run_self_play_process
from src.cluster.TrainerProcess import run_trainer_process
from src.util.tensorboard import TensorboardWriter, log_scalar, log_scalars


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

        self.self_play_processes: list[Process] = []
        self.commander_self_play_pipes: list[PipeConnection] = []
        self.commander_self_play_pipes_by_device: dict[int, list[PipeConnection]] = {}

        self.trainer_device_id = torch.cuda.device_count() - 1 if USE_GPU else 0

    def _setup_connections(self) -> None:
        # The Trainer and Commander has a Pipe connection
        # Each SelfPlay and InferenceServer has a Pipe connection to the LoadBalancer
        # The Commander has a Pipe connection to each SelfPlay and InferenceServer

        # Trainer and Commander
        trainer_commander_pipe, self.commander_trainer_pipe = Pipe(duplex=True)

        self.trainer_process = Process(
            target=run_trainer_process, args=(self.run_id, self.args, trainer_commander_pipe, self.trainer_device_id)
        )
        self.trainer_process.start()

        self.commander_self_play_pipes: list[PipeConnection] = []
        for node_id in range(self.args.cluster.num_self_play_nodes_on_cluster):
            device_id = _get_device_id(node_id, self.args.cluster.num_self_play_nodes_on_cluster)

            self_play_commander_pipe, commander_self_play_pipe = Pipe(duplex=False)
            self.commander_self_play_pipes.append(commander_self_play_pipe)
            self.commander_self_play_pipes_by_device.setdefault(device_id, []).append(commander_self_play_pipe)

            process = Process(
                target=run_self_play_process,
                args=(self.run_id, self.args, self_play_commander_pipe, device_id),
            )
            process.start()
            self.self_play_processes.append(process)

    def run(self) -> Generator[tuple[int, TrainingStats], None, None]:
        """The main loop of the CommanderProcess. The resulting generator yields after each iteration. If the Generator is not consumed, no further iterations will be trained."""

        Path(self.args.save_path).mkdir(parents=True, exist_ok=True)

        log('Setting up connections...')
        self._setup_connections()
        log('Connections set up.')

        # Start CPU usage logger for one SelfPlay process
        self.commander_self_play_pipes[0].send(f'START USAGE LOGGER:{self.run_id}')

        starting_iteration = get_latest_model_iteration(self.args.save_path)
        log(f'Starting training at iteration {starting_iteration}.')

        self._ensure_model_exists(starting_iteration)

        current_best_iteration = starting_iteration

        for pipe in self.commander_self_play_pipes:
            with log_exceptions('SelfPlay setup'):
                pipe.send(f'LOAD MODEL: {current_best_iteration}')
                pipe.send(f'START AT ITERATION: {starting_iteration}')

        last_train_stats = TrainingStats(
            policy_loss=0.0,
            value_loss=0.0,
            total_loss=0.0,
            value_mean=0.0,
            value_std=1.0,  # to avoid early stopping
            grad_norm=0.0,
        )

        with log_exceptions('Commander process'):
            for iteration in range(starting_iteration, self.args.num_iterations):
                # send START AT ITERATION: iteration to Trainer and InferenceServers and SelfPlayers
                print(f'Starting iteration {iteration}.')
                self.commander_trainer_pipe.send(f'START AT ITERATION: {iteration}')
                for pipe in self.commander_self_play_pipes:
                    with log_exceptions('SelfPlay setup'):
                        pipe.send(f'START AT ITERATION: {iteration}')

                # Wait for Trainer to finish
                train_stats: TrainingStats = self.commander_trainer_pipe.recv()  # type: ignore
                assert self.commander_trainer_pipe.recv() == 'FINISHED'
                yield iteration, train_stats

                if train_stats.value_std < 0.01 and last_train_stats.value_std < 0.01:
                    log('Training stopped early due to low value std deviation.')
                    exit()

                last_train_stats = train_stats

                # gating
                with TensorboardWriter(self.run_id, 'gating', postfix_pid=False):
                    # TODO only stop processes on the gating device and make the portion a hyperparameter
                    for pipe in self.commander_self_play_pipes_by_device[self.trainer_device_id][::2]:
                        # only stop half of the self-play processes for gating
                        pipe.send('STOP SELF PLAY')

                    # TODO gating params into args
                    gating_evaluation = ModelEvaluation(
                        iteration + 1,
                        self.args,
                        device_id=self.trainer_device_id,
                        num_games=100,
                        num_searches_per_turn=200,
                    )
                    results = gating_evaluation.play_two_models_search(
                        model_save_path(current_best_iteration, self.args.save_path)
                    )

                    log_scalars(
                        'gating/gating',
                        {
                            'wins': results.wins,
                            'losses': results.losses,
                            'draws': results.draws,
                        },
                        iteration,
                    )

                    result_score = (results.wins + results.draws * 0.5) / gating_evaluation.num_games
                    # win rate with draws ignored
                    result_score = (
                        results.wins / (results.wins + results.losses) if results.wins + results.losses > 0 else 0.0
                    )
                    log(f'Gating evaluation at iteration {iteration} resulted in {result_score} score ({results}).')
                    # TODO make this a parameter in args
                    if result_score > 0.53:  # 55% win rate
                        log(f'Gating evaluation passed at iteration {iteration}.')
                        current_best_iteration = iteration + 1
                        for pipe in self.commander_self_play_pipes:
                            pipe.send(f'LOAD MODEL: {current_best_iteration}')
                    else:
                        log(
                            f'Gating evaluation failed at iteration {iteration}. Keeping current best model {current_best_iteration}.'
                        )

                    log_scalar('gating/current_best_iteration', current_best_iteration, iteration)
                    log_scalar('gating/gating_score', result_score, iteration)

                # start EvaluationProcess
                p = Process(target=run_evaluation_process, args=(self.run_id, self.args, iteration + 1))
                p.start()
                print(f'Started evaluation process for iteration {iteration}.')
                # p.join()
                # print(f'Finished evaluation process for iteration {iteration}.')

        log('Training complete. Sending STOP to all processes.')
        for pipe in self._all_pipes():
            try:
                pipe.send('STOP')
            except BrokenPipeError:
                pass

        self.trainer_process.kill()
        for process in self.self_play_processes:
            process.join(timeout=10)
        exit()

    def _ensure_model_exists(self, starting_iteration: int) -> None:
        model, optimizer = load_model_and_optimizer(
            starting_iteration,
            self.args.network,
            torch.device('cuda' if USE_GPU else 'cpu'),
            self.args.save_path,
            self.args.training.optimizer,
        )
        save_model_and_optimizer(model, optimizer, starting_iteration, self.args.save_path)

    def _all_processes(self) -> list[Process]:
        return self.self_play_processes + [self.trainer_process]

    def _all_pipes(self) -> list[PipeConnection]:
        return self.commander_self_play_pipes + [self.commander_trainer_pipe]


def _get_device_id(i: int, total: int, num_devices: int = torch.cuda.device_count()) -> int:
    # device 0 should have only half the processes of the other devices as device 0 is 50% occupied by the Trainer
    if not USE_GPU:
        return 0

    if num_devices == 1:
        log('Warning: Only one device available. Using device 0.')
        return 0

    assert num_devices > 1, 'There must be at least 2 devices to distribute the processes.'

    num_on_each_device = total / num_devices
    num_on_last_device = round((num_on_each_device) / 2)

    if i < num_on_last_device:
        return torch.cuda.device_count() - 1

    device_id = (i - num_on_last_device) % (num_devices - 1)
    return device_id
