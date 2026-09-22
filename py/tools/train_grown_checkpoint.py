"""Train a grown checkpoint for a fixed number of epochs over the live replay window.

This is the step between growing a model and promoting it: the grown network starts as an exact
copy of its parent's function, and this is where the capacity it was given actually learns. It
trains at the active model's own learning rate, not at a catch-up rate, because the elevated rate
a from-scratch candidate needs would destroy the initialisation on its first few steps.

Sampling is uniform over the live window rather than the production surprise sampler, so an epoch
means what it says: every row presented once per epoch in expectation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.replay.batch_loader import build_training_batch
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.checkpoint.contracts import read_checkpoint_manifest
from src.training.checkpoint.persistence import create_model, create_optimizer
from src.training.network import NetworkParams
from src.training.progressive import ProgressiveModelSizingConfiguration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-config', required=True, type=Path)
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--run-state', required=True, type=Path)
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--state-dict', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--epochs', default=1.0, type=float)
    parser.add_argument('--batch-size', default=2048, type=int)
    parser.add_argument('--learning-rate', required=True, type=float)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--random-seed', default=20260922, type=int)
    parser.add_argument('--report-interval', default=200, type=int)
    arguments = parser.parse_args()

    experiment = load_experiment_configuration(arguments.experiment_config)
    if not isinstance(experiment, ChessExperimentConfiguration):
        raise ValueError('Training a grown checkpoint requires a chess experiment configuration.')
    sizing = experiment.training.progressive_model_sizing
    if not isinstance(sizing, ProgressiveModelSizingConfiguration):
        raise ValueError('Training a grown checkpoint requires a progressive model-sizing configuration.')
    definition = sizing.model(arguments.model_id)
    if not isinstance(definition.network, NetworkParams):
        raise ValueError('Training a grown checkpoint is only defined for convolutional stages.')

    game = ChessImplementation(experiment)
    manifest = read_checkpoint_manifest(arguments.generation, arguments.run_state)
    replay_configuration = experiment.training.lifecycle.replay
    layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=replay_configuration.maximum_policy_entries,
        maximum_legal_actions=game.state.maximum_legal_action_count,
    )
    store = ReplayStore.open(arguments.replay_store, layout, writable=False)
    live_samples = store.state.size
    steps = int(round(arguments.epochs * live_samples / arguments.batch_size))
    device = torch.device('cuda', arguments.device_id)
    torch.manual_seed(arguments.random_seed)
    torch.cuda.manual_seed_all(arguments.random_seed)

    model = create_model(definition.network, device, game.network_dimensions, game.target_layout.auxiliary_heads)
    model.load_state_dict(torch.load(arguments.state_dict, map_location='cpu', weights_only=True))
    model.train()
    optimizer = create_optimizer(model, experiment.training.trainer.optimizer)
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = arguments.learning_rate
    objective = game.training_objective_at(manifest.generation)
    generator = np.random.default_rng(arguments.random_seed)

    print(
        f'live window {live_samples} rows, {arguments.epochs} epoch(s) = {steps} steps at batch {arguments.batch_size}'
    )
    started_at = time.perf_counter()
    for step in range(1, steps + 1):
        indices = generator.integers(0, live_samples, size=arguments.batch_size, dtype=np.int64)
        augmentation = np.zeros(arguments.batch_size, dtype=np.int64)
        batch = build_training_batch(store, game.state, indices, augmentation).to_device(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = objective.calculate_loss(model.training_output(batch.states), batch)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.training.trainer.max_grad_norm)
        optimizer.step()
        if step % arguments.report_interval == 0 or step == steps:
            elapsed = time.perf_counter() - started_at
            print(
                f'step {step}/{steps}  loss {float(loss.total.detach()):.4f}  '
                f'{step * arguments.batch_size / elapsed:.0f} samples/s  {elapsed:.0f}s'
            )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.cpu() for name, value in model.state_dict().items()}, arguments.output)
    store.close()
    print(f'wrote {arguments.output}')


if __name__ == '__main__':
    main()
