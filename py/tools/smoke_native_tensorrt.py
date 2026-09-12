from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from AlphaZeroCpp import (
    AnalysisMode,
    AnalysisParameters,
    BatchedInferenceParameters,
    ChessAnalysis,
    InferenceBackend,
    InferenceConfiguration,
    InferenceDevice,
)

STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


@dataclass(frozen=True)
class PositionComparison:
    moves: tuple[str, ...]
    maximum_policy_difference: float
    maximum_wdl_difference: float
    same_top_action: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare native TorchScript and TensorRT chess inference.')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-backend', choices=('torchscript', 'tensorrt'), default='torchscript')
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=320)
    return parser.parse_args()


def create_analysis(
    model_path: Path,
    backend: InferenceBackend,
    device: int,
    batch_size: int,
) -> ChessAnalysis:
    return ChessAnalysis(
        InferenceConfiguration(
            device_id=device,
            model_path=str(model_path),
            device=InferenceDevice.CUDA,
            backend=backend,
        ),
        AnalysisParameters(1, 1.5, BatchedInferenceParameters(1, batch_size, 2)),
    )


def compare_position(
    reference_analysis: ChessAnalysis,
    tensor_rt_analysis: ChessAnalysis,
    moves: tuple[str, ...],
) -> PositionComparison:
    torch_result = reference_analysis.new_session(STARTING_FEN, list(moves)).analyze(AnalysisMode.POLICY)
    tensor_rt_result = tensor_rt_analysis.new_session(STARTING_FEN, list(moves)).analyze(AnalysisMode.POLICY)
    torch_policy = {candidate.move_uci: candidate.policy_prior for candidate in torch_result.candidates}
    tensor_rt_policy = {candidate.move_uci: candidate.policy_prior for candidate in tensor_rt_result.candidates}
    assert torch_policy.keys() == tensor_rt_policy.keys()
    assert torch_result.outcome is not None
    assert tensor_rt_result.outcome is not None
    maximum_policy_difference = max(abs(torch_policy[action] - tensor_rt_policy[action]) for action in torch_policy)
    maximum_wdl_difference = max(
        abs(torch_value - tensor_rt_value)
        for torch_value, tensor_rt_value in zip(
            (torch_result.outcome.win, torch_result.outcome.draw, torch_result.outcome.loss),
            (tensor_rt_result.outcome.win, tensor_rt_result.outcome.draw, tensor_rt_result.outcome.loss),
            strict=True,
        )
    )
    return PositionComparison(
        moves=moves,
        maximum_policy_difference=maximum_policy_difference,
        maximum_wdl_difference=maximum_wdl_difference,
        same_top_action=torch_result.chosen_move_uci == tensor_rt_result.chosen_move_uci,
    )


def main() -> None:
    arguments = parse_arguments()
    model_backend = (
        InferenceBackend.TORCHSCRIPT if arguments.model_backend == 'torchscript' else InferenceBackend.TENSORRT
    )
    reference_analysis = create_analysis(arguments.model, model_backend, arguments.device, arguments.batch_size)
    tensor_rt_analysis = create_analysis(
        arguments.engine, InferenceBackend.TENSORRT, arguments.device, arguments.batch_size
    )
    positions = (
        (),
        ('e2e4',),
        ('d2d4', 'g8f6'),
        ('e2e4', 'e7e5', 'g1f3', 'b8c6'),
        ('d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3'),
    )
    comparisons = [compare_position(reference_analysis, tensor_rt_analysis, moves) for moves in positions]
    for comparison in comparisons:
        print(comparison)
    print(f'all_top_actions_match={all(item.same_top_action for item in comparisons)}')
    print(f'maximum_policy_difference={max(item.maximum_policy_difference for item in comparisons):.8f}')
    print(f'maximum_wdl_difference={max(item.maximum_wdl_difference for item in comparisons):.8f}')


if __name__ == '__main__':
    main()
