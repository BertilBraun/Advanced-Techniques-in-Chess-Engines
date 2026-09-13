from __future__ import annotations

import argparse
from pathlib import Path

import torch
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.training.network import Network
from src.training.quantization.configuration import QatCheckpointPhase, TensorRtInt8QatConfiguration
from src.training.quantization.runtime import (
    configure_qat,
    export_qat_onnx,
    fixed_batch_example_states,
    fold_scaled_post_activation_batch_norm,
)
from torch import nn


def _seed_refittable_batch_norm_parameters(model: Network) -> None:
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if module.weight is None or module.bias is None:
                raise ValueError('TensorRT QAT template export requires affine BatchNorm parameters.')
            module.weight.fill_(0.875)
            module.bias.fill_(0.03125)
            module.running_mean.fill_(0.0625)
            module.running_var.fill_(1.125)


def export_template(
    configuration_path: Path,
    model_id: str,
    phase: QatCheckpointPhase,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    configuration = load_experiment_configuration(configuration_path)
    if not isinstance(configuration, ChessExperimentConfiguration):
        raise ValueError('QAT TensorRT template export currently supports chess experiments.')
    quantization = configuration.training.trainer.quantization
    if not isinstance(quantization, TensorRtInt8QatConfiguration):
        raise ValueError('The experiment must configure TensorRT INT8 QAT.')
    matching_models = tuple(
        model for model in configuration.training.progressive_model_sizing.models if model.model_id == model_id
    )
    if len(matching_models) != 1:
        raise ValueError(f'Model ID must identify exactly one configured model: {model_id}')

    torch.manual_seed(configuration.training.random_seed)
    model = Network(matching_models[0].network, device, configuration.network_dimensions).to(device)
    _seed_refittable_batch_norm_parameters(model)
    calibration_states = torch.randint(
        0,
        2,
        (
            quantization.calibration_positions,
            configuration.network_dimensions.channels,
            configuration.network_dimensions.rows,
            configuration.network_dimensions.columns,
        ),
        device=device,
        dtype=torch.float32,
    )

    def calibrate(candidate: torch.nn.Module) -> None:
        candidate(calibration_states)

    model = configure_qat(model, calibrate)
    if phase is QatCheckpointPhase.DEPLOYMENT:
        fold_scaled_post_activation_batch_norm(model)
    example_states = fixed_batch_example_states(calibration_states, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_qat_onnx(model, output_path, example_states)


def main() -> None:
    parser = argparse.ArgumentParser(description='Export a deterministic QAT ONNX graph for TensorRT templating.')
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--phase', type=QatCheckpointPhase, choices=tuple(QatCheckpointPhase), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    export_template(
        arguments.configuration,
        arguments.model_id,
        arguments.phase,
        arguments.output,
        arguments.batch_size,
        torch.device(arguments.device),
    )


if __name__ == '__main__':
    main()
