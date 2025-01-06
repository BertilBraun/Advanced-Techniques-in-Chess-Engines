from multiprocessing import Process
from multiprocessing.connection import Pipe, PipeConnection
from pathlib import Path

from src.util.save_paths import get_latest_model_iteration
from src.cluster.EvaluationProcess import run_evaluation_process
from src.cluster.InferenceServerProcess import run_inference_server
from src.cluster.LoadBalanceProcess import run_load_balancer_process
from src.cluster.SelfPlayProcess import run_self_play_process
from src.cluster.TrainerProcess import run_trainer_process
from src.settings import TRAINING_ARGS
from src.util.log import log

# setup one Trainer Process
# setup one LoadBalancer Process
# setup num_gpus InferenceServer Processes
# setup num_self_play_nodes_on_cluster SelfPlay Processes
# setup command process, which notifies InferenceServers to load new models and notifies SelfPlay Processes to write to the next iteration


class CommanderProcess:
    def __init__(self, num_self_play_nodes: int, num_inference_nodes: int) -> None:
        self.num_self_play_nodes = num_self_play_nodes
        self.num_inference_nodes = num_inference_nodes

        self.trainer_process: Process
        self.commander_trainer_pipe: PipeConnection

        self.load_balancer_process: Process

        self.inference_server_processes: list[Process] = []
        self.commander_inference_server_pipes: list[PipeConnection] = []

        self.self_play_processes: list[Process] = []
        self.commander_self_play_pipes: list[PipeConnection] = []

    def _setup_connections(self) -> None:
        # The Trainer and Commander has a Pipe connection
        # Each SelfPlay and InferenceServer has a Pipe connection to the LoadBalancer
        # The Commander has a Pipe connection to each SelfPlay and InferenceServer

        # Trainer and Commander
        trainer_device_id = 0
        trainer_commander_pipe, self.commander_trainer_pipe = Pipe(duplex=True)

        self.trainer_process = Process(target=run_trainer_process, args=(trainer_commander_pipe, trainer_device_id))
        self.trainer_process.start()

        # SelfPlay and LoadBalancer
        self_play_to_load_balancer_pipes: list[PipeConnection] = []
        load_balancer_input_pipes: list[PipeConnection] = []
        for _ in range(self.num_self_play_nodes):
            self_play_pipe, load_balancer_input_pipe = Pipe(duplex=True)
            self_play_to_load_balancer_pipes.append(self_play_pipe)
            load_balancer_input_pipes.append(load_balancer_input_pipe)

        # InferenceServer and LoadBalancer
        inference_server_to_load_balancer_pipes: list[PipeConnection] = []
        load_balancer_output_pipes: list[PipeConnection] = []
        for _ in range(self.num_inference_nodes):
            inference_server_pipe, load_balancer_output_pipe = Pipe(duplex=True)
            inference_server_to_load_balancer_pipes.append(inference_server_pipe)
            load_balancer_output_pipes.append(load_balancer_output_pipe)

        self.load_balancer_process = Process(
            target=run_load_balancer_process, args=(load_balancer_input_pipes, load_balancer_output_pipes)
        )
        self.load_balancer_process.start()

        self.commander_inference_server_pipes: list[PipeConnection] = []
        for device_id in range(self.num_inference_nodes):
            inference_server_commander_pipe, commander_inference_server_pipe = Pipe(duplex=False)
            self.commander_inference_server_pipes.append(commander_inference_server_pipe)

            p = Process(
                target=run_inference_server,
                args=(
                    inference_server_to_load_balancer_pipes[device_id],
                    inference_server_commander_pipe,
                    device_id + 1,
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
                args=(self_play_commander_pipe, self_play_to_load_balancer_pipes[client_idx]),
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

        for iteration in range(starting_iteration, TRAINING_ARGS.num_iterations):
            # send START AT ITERATION: iteration to Trainer and InferenceServers and SelfPlayers
            for pipe in (
                self.commander_inference_server_pipes + self.commander_self_play_pipes + [self.commander_trainer_pipe]
            ):
                pipe.send(f'START AT ITERATION: {iteration}')
            log(f'All processes started at iteration {iteration}.')

            # Wait for Trainer to finish
            assert self.commander_trainer_pipe.recv() == 'FINISHED'
            log(f'Trainer finished at iteration {iteration}.')

            # start EvaluationProcess
            p = Process(
                target=run_evaluation_process,
                args=(
                    0,  # let the evaluation process run on the trainer device
                    iteration,
                ),
            )
            p.start()

        log('Training complete. Sending STOP to all processes.')
        for pipe in (
            self.commander_inference_server_pipes + self.commander_self_play_pipes + [self.commander_trainer_pipe]
        ):
            pipe.send('STOP')
