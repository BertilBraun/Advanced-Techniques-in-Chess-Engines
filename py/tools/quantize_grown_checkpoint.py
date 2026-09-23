"""Turn float weights into a QAT checkpoint the production trainer can resume from.

A grown model cannot inherit its parent's quantiser state, because that structure follows the depth
and width that just changed, so it has to be wrapped and calibrated from scratch. This is the same
wrap-and-calibrate the trainer performs for a model it bootstraps, applied to weights that already
exist rather than to a random initialisation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from src.evaluation.dataset import load_dataset_probe_states
from src.evaluation.process import resolve_project_path
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.training.checkpoint.paths import qat_state_save_path
from src.training.checkpoint.persistence import create_model, create_optimizer
from src.training.progressive import ProgressiveModelSizingConfiguration
from src.training.quantization.checkpoint import save_qat_model_and_optimizer
from src.training.quantization.configuration import TensorRtInt8QatConfiguration
from src.training.quantization.runtime import configure_qat, save_qat_state


def _calibration_loop(states: torch.Tensor, batch_size: int):
    def run(model: torch.nn.Module) -> None:
        was_training = model.training
        model.eval()
        with torch.inference_mode():
            for batch in states.split(batch_size):
                model(batch)
        model.train(was_training)

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-config', required=True, type=Path)
    parser.add_argument('--state-dict', required=True, type=Path)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--output-run-state', required=True, type=Path)
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--device-id', default=0, type=int)
    arguments = parser.parse_args()

    experiment = load_experiment_configuration(arguments.experiment_config)
    if not isinstance(experiment, ChessExperimentConfiguration):
        raise ValueError('Quantizing a grown checkpoint requires a chess experiment configuration.')
    sizing = experiment.training.progressive_model_sizing
    if not isinstance(sizing, ProgressiveModelSizingConfiguration):
        raise ValueError('Quantizing a grown checkpoint requires a progressive model-sizing configuration.')
    quantization = experiment.training.trainer.quantization
    if not isinstance(quantization, TensorRtInt8QatConfiguration):
        raise ValueError('Quantizing a grown checkpoint requires the INT8 QAT trainer configuration.')
    definition = sizing.model(arguments.model_id)

    game = ChessImplementation(experiment)
    device = torch.device('cuda', arguments.device_id)
    model = create_model(definition.network, device, game.network_dimensions, game.target_layout.auxiliary_heads)
    weights = torch.load(arguments.state_dict, map_location=device, weights_only=True)
    model.load_state_dict({name: weights[name] for name in model.state_dict()})

    calibration_states = load_dataset_probe_states(
        resolve_project_path(experiment.evaluation.dataset.path),
        game.state,
        quantization.calibration_positions,
    ).to(device=device, dtype=torch.float32)
    model = configure_qat(
        model,
        _calibration_loop(calibration_states, game.self_play_configuration.inference.inference_batch_size),
    )
    optimizer = create_optimizer(model, experiment.training.trainer.optimizer)

    run_state = arguments.output_run_state
    run_state.mkdir(parents=True, exist_ok=True)
    generation = arguments.generation
    qat_identity = save_qat_state(model, qat_state_save_path(generation, run_state), 0)
    reference = save_qat_model_and_optimizer(
        model,
        optimizer,
        generation,
        0,
        run_state,
        qat_identity,
        calibration_states,
        quantization_configuration=quantization,
        bootstrap_policy_prior=None,
    )
    print(f'wrote QAT checkpoint {generation} to {run_state}')
    print(f'  inference artifact {reference.inference_model_path.name}')
    print(f'  qat state {qat_state_save_path(generation, run_state).name}')


if __name__ == '__main__':
    main()
