from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns, time_ns

import chess
import numpy as np
import torch
from AlphaZeroCpp import ChessPosition
from pydantic import BaseModel, ConfigDict
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.representation import decode_packed_planes
from src.util.atomic_file import write_text_atomically
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision_if_available


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    repeat: int
    simulations: int
    root_visits: int
    inference_calls: int
    inference_positions: int
    elapsed_seconds: float
    simulations_per_second: float
    inference_calls_per_second: float
    selected_move_uci: str


class BenchmarkProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    command: tuple[str, ...]
    created_unix_nanoseconds: int
    source_revision: str
    source_dirty: bool | None
    model_path: str
    model_sha256: str
    starting_fen: str
    device: str
    cuda_visible_devices: str | None
    python_version: str
    python_chess_version: str
    torch_version: str
    torch_cuda_version: str | None
    cudnn_version: int | None
    platform: str
    processor: str
    logical_cpu_count: int | None


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    implementation: str = 'naive-python-chess-puct'
    game_batch_size: int = 1
    inference_batch_size: int = 1
    parallel_searches: int = 1
    exploration_constant: float
    root_initialization_inference_calls: int
    warmup_simulations: int
    warmup_inference_calls: int
    provenance: BenchmarkProvenance
    runs: tuple[BenchmarkRun, ...]


@dataclass
class SearchNode:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, SearchNode] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class Evaluation:
    priors: tuple[tuple[chess.Move, float], ...]
    value: float


class BatchOneEvaluator:
    def __init__(self, model_path: Path, device: torch.device) -> None:
        self._device = device
        self._model = torch.jit.load(str(model_path), map_location=device)
        self._model.eval()
        self.inference_calls = 0
        self.inference_positions = 0

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> Evaluation:
        native_position = ChessPosition(board.fen())
        representation = CHESS_STATE_CONTRACT.representation
        encoded = CHESS_STATE_CONTRACT.encode_network_input(native_position)
        decoded = decode_packed_planes(
            encoded,
            representation.packed_planes,
            representation.binary_channels,
            representation.scalar_channels,
        )
        inputs = torch.from_numpy(decoded).unsqueeze(0).to(self._device, dtype=torch.float32)
        policy, wdl, _search_correction = self._model(inputs)
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
        self.inference_calls += 1
        self.inference_positions += 1

        legal_moves = tuple(board.legal_moves)
        action_ids = tuple(native_position.action_id_from_uci(move.uci()) for move in legal_moves)
        legal_logits = policy[0, list(action_ids)].to(device='cpu', dtype=torch.float64)
        normalized = torch.softmax(legal_logits, dim=0).numpy()
        if not np.isfinite(normalized).all():
            raise ValueError('Model returned invalid logits for legal chess moves.')
        outcome = wdl[0].to(device='cpu', dtype=torch.float64)
        value = float(outcome[0] - outcome[2])
        return Evaluation(
            priors=tuple((move, float(prior)) for move, prior in zip(legal_moves, normalized, strict=True)),
            value=value,
        )


class NaivePythonMcts:
    def __init__(self, board: chess.Board, evaluator: BatchOneEvaluator, exploration_constant: float) -> None:
        self._board = board.copy(stack=True)
        self._evaluator = evaluator
        self._exploration_constant = exploration_constant
        self.root = SearchNode(prior=1.0)
        self._expand(self.root, self._evaluator.evaluate(self._board))

    def run_simulation(self) -> None:
        board = self._board.copy(stack=True)
        node = self.root
        path = [node]
        while node.children:
            move, node = select_child(node, self._exploration_constant)
            board.push(move)
            path.append(node)

        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            evaluation = self._evaluator.evaluate(board)
            self._expand(node, evaluation)
            value = evaluation.value
        else:
            value = terminal_value(outcome, board.turn)
        backup(path, value)

    @staticmethod
    def _expand(node: SearchNode, evaluation: Evaluation) -> None:
        assert not node.children
        node.children = {move: SearchNode(prior=prior) for move, prior in evaluation.priors}

    def selected_move(self) -> chess.Move:
        if not self.root.children:
            raise ValueError('Cannot select a move from an unexpanded root.')
        return max(self.root.children.items(), key=lambda item: (item[1].visits, item[1].prior))[0]


def select_child(parent: SearchNode, exploration_constant: float) -> tuple[chess.Move, SearchNode]:
    parent_scale = max(1, parent.visits) ** 0.5

    def score(item: tuple[chess.Move, SearchNode]) -> float:
        _, child = item
        exploitation = -child.mean_value
        exploration = exploration_constant * child.prior * parent_scale / (1 + child.visits)
        return exploitation + exploration

    return max(parent.children.items(), key=score)


def terminal_value(outcome: chess.Outcome, player_to_move: chess.Color) -> float:
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == player_to_move else -1.0


def backup(path: list[SearchNode], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value


def _provenance(arguments: argparse.Namespace, model_path: Path, device: torch.device) -> BenchmarkProvenance:
    detected = read_source_revision_if_available()
    return BenchmarkProvenance(
        command=tuple(sys.argv),
        created_unix_nanoseconds=time_ns(),
        source_revision=arguments.source_revision or (detected.commit if detected is not None else 'unavailable'),
        source_dirty=None if detected is None else detected.dirty,
        model_path=str(model_path.resolve()),
        model_sha256=file_sha256(model_path),
        starting_fen=arguments.fen,
        device=str(device),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        python_version=sys.version,
        python_chess_version=chess.__version__,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        platform=platform.platform(),
        processor=platform.processor(),
        logical_cpu_count=os.cpu_count(),
    )


def _run_once(
    evaluator: BatchOneEvaluator,
    board: chess.Board,
    exploration_constant: float,
    simulations: int,
    repeat: int,
) -> tuple[BenchmarkRun, int]:
    calls_before_initialization = evaluator.inference_calls
    search = NaivePythonMcts(board, evaluator, exploration_constant)
    initialization_calls = evaluator.inference_calls - calls_before_initialization
    calls_before = evaluator.inference_calls
    started = perf_counter_ns()
    for _ in range(simulations):
        search.run_simulation()
    elapsed_seconds = (perf_counter_ns() - started) / 1_000_000_000
    measured_calls = evaluator.inference_calls - calls_before
    return (
        BenchmarkRun(
            repeat=repeat,
            simulations=simulations,
            root_visits=search.root.visits,
            inference_calls=measured_calls,
            inference_positions=measured_calls,
            elapsed_seconds=elapsed_seconds,
            simulations_per_second=simulations / elapsed_seconds,
            inference_calls_per_second=measured_calls / elapsed_seconds,
            selected_move_uci=search.selected_move().uci(),
        ),
        initialization_calls,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark deliberately naive Python AlphaZero-style chess MCTS.')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fen', default=chess.STARTING_FEN)
    parser.add_argument('--simulations', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup-simulations', type=int, default=32)
    parser.add_argument('--exploration-constant', type=float, default=1.0)
    parser.add_argument('--source-revision')
    arguments = parser.parse_args()
    if arguments.simulations <= 0 or arguments.repeats <= 0 or arguments.warmup_simulations < 0:
        raise ValueError('Simulation and repeat counts must be positive; warm-up count must be nonnegative.')
    if arguments.exploration_constant <= 0.0:
        raise ValueError('Exploration constant must be positive.')
    if not arguments.model.is_file():
        raise ValueError(f'Model does not exist: {arguments.model}')
    chess.Board(arguments.fen)
    return arguments


def main() -> None:
    arguments = parse_arguments()
    device = torch.device(arguments.device)
    board = chess.Board(arguments.fen)
    warmup_evaluator = BatchOneEvaluator(arguments.model, device)
    warmup = NaivePythonMcts(board, warmup_evaluator, arguments.exploration_constant)
    for _ in range(arguments.warmup_simulations):
        warmup.run_simulation()
    warmup_inference_calls = warmup_evaluator.inference_calls

    runs: list[BenchmarkRun] = []
    root_initialization_calls = 0
    for repeat in range(1, arguments.repeats + 1):
        run, initialization_calls = _run_once(
            warmup_evaluator,
            board,
            arguments.exploration_constant,
            arguments.simulations,
            repeat,
        )
        runs.append(run)
        root_initialization_calls += initialization_calls
    result = BenchmarkResult(
        exploration_constant=arguments.exploration_constant,
        root_initialization_inference_calls=root_initialization_calls,
        warmup_simulations=arguments.warmup_simulations,
        warmup_inference_calls=warmup_inference_calls,
        provenance=_provenance(arguments, arguments.model, device),
        runs=tuple(runs),
    )
    write_text_atomically(arguments.output, result.model_dump_json(indent=2) + '\n')
    print(json.dumps(result.model_dump(mode='json'), indent=2))


if __name__ == '__main__':
    main()
