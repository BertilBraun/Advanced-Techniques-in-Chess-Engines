from __future__ import annotations

import multiprocessing
import socket
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess

from src.experiment.configuration import ExperimentConfiguration
from src.games.implementation import GameImplementation
from src.training.checkpoint import CheckpointReference
from src.training.progress import TrainingProgress
from src.training.trainer.contracts import (
    RankTrainingFailure,
    RankTrainingResult,
    ResolvedTrainingParameters,
    StopTrainerCommand,
    TrainerQuantum,
    TrainerResponse,
    TrainerStartup,
    TrainerStopped,
    TrainingQuantumResult,
    TrainingStatistics,
    TrainQuantumCommand,
)
from src.training.trainer.rank import trainer_rank_main


class TrainerGroup:
    def __init__(
        self,
        configuration: ExperimentConfiguration,
        game: GameImplementation,
        startup: TrainerStartup,
    ) -> None:
        self.configuration = configuration
        self.game = game
        topology = configuration.training.topology.trainer
        self.world_size = len(topology.ddp_device_ids)
        self._closed = False
        context = multiprocessing.get_context('spawn')
        rendezvous_port = _available_tcp_port()
        self._connections: list[Connection] = []
        self._processes: list[BaseProcess] = []
        for rank in range(self.world_size):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=trainer_rank_main,
                args=(
                    child,
                    rank,
                    self.world_size,
                    rendezvous_port,
                    configuration.model_dump_json(),
                    startup,
                ),
                name=f'trainer-rank-{rank}',
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def train_quantum(
        self,
        quantum: TrainerQuantum,
    ) -> TrainingQuantumResult:
        self._ensure_open()
        source_generation = quantum.model_progress.model_generation
        target_progress = quantum.model_progress.next_generation
        comparable_generation = quantum.replay_source_progress.model_generation
        command = TrainQuantumCommand(
            replay=quantum.replay,
            source_progress=quantum.model_progress,
            target_progress=target_progress,
            parameters=ResolvedTrainingParameters(
                learning_rate=self.configuration.training.trainer.learning_rate.value_at(source_generation),
                objective=self.game.training_objective_at(comparable_generation),
            ),
            replay_source_optimizer_steps=quantum.replay_source_progress.completed_optimizer_steps,
        )
        for connection in self._connections:
            connection.send(command)
        results = self._receive_quantum_responses()
        checkpoint = self._validate_quantum_results(results, target_progress)
        return TrainingQuantumResult(
            completed_optimizer_steps=target_progress.completed_optimizer_steps,
            checkpoint=checkpoint,
            statistics=self._aggregate_statistics(results),
        )

    def _validate_quantum_results(
        self,
        results: tuple[RankTrainingResult, ...],
        target_progress: TrainingProgress,
    ) -> CheckpointReference:
        if len(results) != self.world_size:
            raise RuntimeError('Trainer ranks returned an invalid response set.')
        expected_steps = target_progress.completed_optimizer_steps
        if any(result.completed_optimizer_steps != expected_steps for result in results):
            raise RuntimeError('Trainer ranks disagree about completed optimizer steps.')
        rank_zero = results[0]
        if rank_zero.rank != 0 or rank_zero.checkpoint is None:
            raise RuntimeError('Rank zero did not return the completed checkpoint.')
        if any(result.checkpoint is not None for result in results[1:]):
            raise RuntimeError('Only rank zero may return a checkpoint.')
        if rank_zero.distributions is None or any(result.distributions is not None for result in results[1:]):
            raise RuntimeError('Only rank zero must return training distributions.')
        auxiliary_count = len(rank_zero.auxiliary_losses)
        if any(len(result.auxiliary_losses) != auxiliary_count for result in results):
            raise RuntimeError('Trainer ranks disagree about auxiliary statistics.')
        return rank_zero.checkpoint

    def _aggregate_statistics(self, results: tuple[RankTrainingResult, ...]) -> TrainingStatistics:
        auxiliary_count = len(results[0].auxiliary_losses)
        distributions = results[0].distributions
        assert distributions is not None
        return TrainingStatistics(
            policy_loss=_mean(tuple(result.policy_loss for result in results)),
            wdl_loss=_mean(tuple(result.wdl_loss for result in results)),
            auxiliary_losses=tuple(
                _mean(tuple(result.auxiliary_losses[index] for result in results)) for index in range(auxiliary_count)
            ),
            total_loss=_mean(tuple(result.total_loss for result in results)),
            gradient_norm=_mean(tuple(result.gradient_norm for result in results)),
            term_trunk_gradients=tuple(
                _mean(tuple(result.term_trunk_gradients[index] for result in results))
                for index in range(len(results[0].term_trunk_gradients))
            ),
            training_samples_per_second=(
                self.configuration.training.trainer.global_batch_size
                * self.configuration.training.lifecycle.credit.optimizer_steps_per_quantum
                / max(result.elapsed_seconds for result in results)
            ),
            elapsed_seconds=max(result.elapsed_seconds for result in results),
            distributions=distributions,
        )

    def close(self) -> None:
        if self._closed:
            return
        for connection in self._connections:
            connection.send(StopTrainerCommand())
        for connection in self._connections:
            response = self._receive(connection)
            match response:
                case TrainerStopped():
                    pass
                case _:
                    raise RuntimeError('Trainer rank did not acknowledge shutdown.')
        for process in self._processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f'Trainer rank exited with code {process.exitcode}.')
        for connection in self._connections:
            connection.close()
        self._closed = True

    def _receive_quantum_responses(self) -> tuple[RankTrainingResult, ...]:
        pending = set(self._connections)
        results: list[RankTrainingResult] = []
        while pending:
            for connection in wait(pending):
                try:
                    response = self._receive(connection)
                except RuntimeError:
                    self._terminate_after_rank_failure()
                    raise
                pending.remove(connection)
                match response:
                    case RankTrainingResult():
                        results.append(response)
                    case RankTrainingFailure():
                        self._terminate_after_rank_failure()
                        raise RuntimeError(f'DDP training quantum failed on rank {response.rank}: {response.error}')
                    case TrainerStopped():
                        self._terminate_after_rank_failure()
                        raise RuntimeError('Trainer rank stopped during a training quantum.')
        return tuple(sorted(results, key=lambda result: result.rank))

    def _terminate_after_rank_failure(self) -> None:
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join()
        for connection in self._connections:
            connection.close()
        self._closed = True

    @staticmethod
    def _receive(connection: Connection) -> TrainerResponse:
        try:
            response: TrainerResponse = connection.recv()
        except EOFError as error:
            raise RuntimeError('Trainer rank connection closed unexpectedly.') from error
        return response

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError('Trainer group is closed.')


def _available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return int(listener.getsockname()[1])


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError('Cannot average an empty value set.')
    return sum(values) / len(values)
