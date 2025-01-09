import torch
from torch.multiprocessing import Process, Pipe, Queue
from pathlib import Path

from src.settings import TRAINING_ARGS
from src.util.log import log
from src.util.PipeConnection import PipeConnection
from src.util.save_paths import get_latest_model_iteration
from src.cluster.EvaluationProcess import run_evaluation_process
from src.cluster.InferenceServerProcess import run_inference_server
from src.cluster.CacheLayerProcess import run_caching_layer
from src.cluster.SelfPlayProcess import run_self_play_process
from src.cluster.TrainerProcess import run_trainer_process

# setup one Trainer Process
# setup one CacheLayer Process
# setup num_inference_nodes InferenceServer Processes
# setup num_self_play_nodes SelfPlay Processes
# setup command process, which notifies InferenceServers to load new models and notifies SelfPlay Processes to write to the next iteration


class CommanderProcess:
    def __init__(self, num_self_play_nodes: int, num_inference_nodes: int) -> None:
        self.num_self_play_nodes = num_self_play_nodes
        self.num_inference_nodes = num_inference_nodes

        self.trainer_process: Process
        self.commander_trainer_pipe: PipeConnection

        self.cache_layer_process: Process
        self.commander_cache_layer_pipe: PipeConnection

        self.inference_server_processes: list[Process] = []
        self.commander_inference_server_pipes: list[PipeConnection] = []

        self.self_play_processes: list[Process] = []
        self.commander_self_play_pipes: list[PipeConnection] = []

        # Keep track of them to solve mutliprocessing spawn issue where they were already freed again
        self.all_queues: list[Queue] = []

    def _setup_connections(self) -> None:
        # The Trainer and Commander has a Pipe connection
        # Each SelfPlay and InferenceServer has a Pipe connection to the LoadBalancer
        # The Commander has a Pipe connection to each SelfPlay and InferenceServer

        # Trainer and Commander
        trainer_device_id = 0
        trainer_commander_pipe, self.commander_trainer_pipe = Pipe(duplex=True)

        self.trainer_process = Process(target=run_trainer_process, args=(trainer_commander_pipe, trainer_device_id))
        self.trainer_process.start()

        cache_layer_input_queue = Queue()
        self_play_response_queues = [Queue() for _ in range(self.num_self_play_nodes)]
        inference_input_queue = Queue()
        cache_layer_response_queue = Queue()

        self.all_queues = [
            cache_layer_input_queue,
            *self_play_response_queues,
            inference_input_queue,
            cache_layer_response_queue,
        ]

        cache_layer_commander_pipe, self.commander_cache_layer_pipe = Pipe(duplex=False)
        self.cache_layer_process = Process(
            target=run_caching_layer,
            args=(
                cache_layer_input_queue,
                self_play_response_queues,
                inference_input_queue,
                cache_layer_response_queue,
                cache_layer_commander_pipe,
            ),
        )
        self.cache_layer_process.start()

        self.commander_inference_server_pipes: list[PipeConnection] = []
        for device_id in range(self.num_inference_nodes):
            inference_server_commander_pipe, commander_inference_server_pipe = Pipe(duplex=False)
            self.commander_inference_server_pipes.append(commander_inference_server_pipe)

            p = Process(
                target=run_inference_server,
                args=(
                    inference_input_queue,
                    cache_layer_response_queue,
                    inference_server_commander_pipe,
                    1 + (device_id % max(torch.cuda.device_count() - 1, 1)),
                ),
            )
            p.start()
            self.inference_server_processes.append(p)

        self.commander_self_play_pipes: list[PipeConnection] = []
        for client_idx in range(self.num_self_play_nodes):
            self_play_commander_pipe, commander_self_play_pipe = Pipe(duplex=False)
            self.commander_self_play_pipes.append(commander_self_play_pipe)

            p = Process(
                target=run_self_play_process,
                args=(
                    self_play_commander_pipe,
                    cache_layer_input_queue,
                    self_play_response_queues[client_idx],
                    client_idx,
                ),
            )
            p.start()
            self.self_play_processes.append(p)

    def run(self):
        Path(TRAINING_ARGS.save_path).mkdir(parents=True, exist_ok=True)

        log('Setting up connections...')
        self._setup_connections()
        log('Connections set up.')

        starting_iteration = get_latest_model_iteration()
        log(f'Starting training at iteration {starting_iteration}.')

        try:
            for iteration in range(starting_iteration, TRAINING_ARGS.num_iterations):
                # send START AT ITERATION: iteration to Trainer and InferenceServers and SelfPlayers
                for pipe in self._all_pipes():
                    pipe.send(f'START AT ITERATION: {iteration}')
                log(f'All processes started at iteration {iteration}.')

                # Wait for Trainer to finish
                assert self.commander_trainer_pipe.recv() == 'FINISHED'
                log(f'Trainer finished at iteration {iteration}.')

                # start EvaluationProcess
                p = Process(target=run_evaluation_process, args=(iteration,))
                p.start()
        finally:
            log('Training complete. Sending STOP to all processes.')
            for pipe in self._all_pipes():
                pipe.send('STOP')

            for process in (
                self.self_play_processes
                + self.inference_server_processes
                + [self.trainer_process, self.cache_layer_process]
            ):
                process.join()

    def _all_pipes(self) -> list[PipeConnection]:
        return (
            self.commander_inference_server_pipes
            + self.commander_self_play_pipes
            + [self.commander_trainer_pipe, self.commander_cache_layer_pipe]
        )
