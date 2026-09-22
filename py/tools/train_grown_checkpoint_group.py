"""Train a grown checkpoint across every trainer GPU, one epoch over the live replay window.

Drives the production TrainerGroup rather than a private loop, so the quantisation, precision,
sampler and data-parallel layout are the ones the run itself uses, and the checkpoints it writes
are ordinary checkpoints. The learning rate is the active model's, not a catch-up rate: the
elevated rate a from-scratch candidate needs would destroy the initialisation it starts from.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.progress import TrainingProgress
from src.training.progressive import ProgressiveModelSizingConfiguration
from src.training.trainer import TrainerGroup
from src.training.trainer.contracts import TrainerQuantum, TrainerStartup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-config', required=True, type=Path)
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--starting-generation', required=True, type=int)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--epochs', default=1.0, type=float)
    parser.add_argument('--learning-rate', required=True, type=float)
    arguments = parser.parse_args()

    experiment = load_experiment_configuration(arguments.experiment_config)
    if not isinstance(experiment, ChessExperimentConfiguration):
        raise ValueError('Training a grown checkpoint requires a chess experiment configuration.')
    sizing = experiment.training.progressive_model_sizing
    if not isinstance(sizing, ProgressiveModelSizingConfiguration):
        raise ValueError('Training a grown checkpoint requires a progressive model-sizing configuration.')
    definition = sizing.model(arguments.model_id)

    game = ChessImplementation(experiment)
    replay_configuration = experiment.training.lifecycle.replay
    layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=replay_configuration.maximum_policy_entries,
        maximum_legal_actions=game.state.maximum_legal_action_count,
    )
    store = ReplayStore.open(arguments.replay_store, layout, writable=False)
    state = store.state
    replay = ReplayDescription(
        path=arguments.replay_store,
        head=state.head,
        size=state.size,
        logical_capacity=state.logical_capacity,
        maximum_capacity=state.maximum_capacity,
        layout=layout,
    )
    store.close()

    credit = experiment.training.lifecycle.credit
    steps_per_quantum = credit.optimizer_steps_per_quantum
    batch = experiment.training.trainer.global_batch_size
    quanta = max(1, int(round(arguments.epochs * state.size / batch / steps_per_quantum)))
    print(
        f'live window {state.size} rows, {arguments.epochs} epoch(s) = '
        f'{quanta} quanta x {steps_per_quantum} steps at batch {batch}',
        flush=True,
    )

    trainer = TrainerGroup(
        experiment,
        game,
        TrainerStartup(
            network=definition.network,
            save_path=arguments.model_path,
            starting_generation=arguments.starting_generation,
        ),
    )
    completed = arguments.starting_generation * steps_per_quantum
    started_at = time.perf_counter()
    try:
        for index in range(quanta):
            progress = TrainingProgress(
                completed_optimizer_steps=completed,
                optimizer_steps_per_generation=steps_per_quantum,
            )
            result = trainer.train_quantum(
                TrainerQuantum(
                    replay=replay,
                    model_progress=progress,
                    replay_source_progress=progress,
                    base_learning_rate=arguments.learning_rate,
                )
            )
            completed = result.completed_optimizer_steps
            elapsed = time.perf_counter() - started_at
            print(
                f'quantum {index + 1}/{quanta}  generation {result.checkpoint.generation}  '
                f'loss {result.statistics.total_loss:.4f}  '
                f'{completed * batch / elapsed:.0f} samples/s  {elapsed:.0f}s',
                flush=True,
            )
    finally:
        trainer.close()
    print(f'final generation {completed // steps_per_quantum} in {arguments.model_path}', flush=True)


if __name__ == '__main__':
    main()
