"""Checks the whole teacher chain against real Lc0 on identical positions.

This is the gate for the teacher experiment. The plane order, the history planes, the 1858-to-1880
policy permutation and the WDL conversion are all asserted at once: if this project's wrapped teacher
reproduces Lc0's own root priors, every one of them is right, and if it does not, none of the match
results afterwards mean anything.

Positions are driven as `position startpos moves ...` on both sides rather than as a FEN, because a
FEN would let Lc0 fill history itself and hide exactly the bug this is looking for.

Requires the native extension and an Lc0 binary, so it runs on the node.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from src.distillation.lc0_teacher import teacher_input_dtype
from src.games.chess.contract import CHESS_STATE_CONTRACT, decode_lc0_planes

MOVE_PRIOR_PATTERN = re.compile(r'^info string\s+(?P<move>[a-h][1-8][a-h][1-8][qrbn]?)\s.*\(P:\s*(?P<prior>[0-9.]+)%\)')
WDL_PATTERN = re.compile(r'\bwdl\s+(\d+)\s+(\d+)\s+(\d+)\b')


@dataclass(frozen=True)
class TeacherReference:
    priors: dict[str, float]
    wdl: tuple[float, float, float] | None


@dataclass(frozen=True)
class PositionComparison:
    moves_uci: tuple[str, ...]
    maximum_policy_difference: float
    top_move_agrees: bool
    wdl_difference: float | None


def read_opening_lines(path: Path, limit: int) -> tuple[tuple[str, ...], ...]:
    lines: list[tuple[str, ...]] = []
    for row in path.read_text(encoding='utf-8').splitlines():
        if row.startswith('#') or not row.strip():
            continue
        columns = row.split('\t')
        if len(columns) < 2:
            continue
        lines.append(tuple(columns[1].split()))
        if len(lines) == limit:
            break
    if not lines:
        raise SystemExit(f'No opening lines parsed from {path}.')
    return tuple(lines)


def query_lc0(
    binary: Path, network: Path, moves_uci: tuple[str, ...], nodes: int, extra_arguments: list[str]
) -> TeacherReference:
    command = [
        str(binary),
        f'--weights={network}',
        '--verbose-move-stats',
        # Lc0's search reports priors after its own policy temperature; the raw network softmax is at 1.
        '--policy-softmax-temp=1.0',
        '--threads=1',
        '--minibatch-size=1',
        *extra_arguments,
    ]
    # `quit` goes only after `bestmove`: sent earlier, Lc0 exits before printing any search output.
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert process.stdin is not None and process.stdout is not None
    output: list[str] = []

    def send(line: str) -> None:
        process.stdin.write(line + '\n')
        process.stdin.flush()

    def read_until(prefix: str) -> None:
        for line in process.stdout:
            output.append(line.rstrip('\n'))
            if line.startswith(prefix):
                return
        raise SystemExit(f'Lc0 exited before printing {prefix!r}.\nLast output:\n' + '\n'.join(output[-40:]))

    send('uci')
    read_until('uciok')
    send('setoption name UCI_ShowWDL value true')
    send('isready')
    read_until('readyok')
    send(f'position startpos moves {" ".join(moves_uci)}')
    send(f'go nodes {nodes}')
    read_until('bestmove')
    send('quit')
    process.wait(timeout=30)

    priors: dict[str, float] = {}
    wdl: tuple[float, float, float] | None = None
    for line in output:
        match = MOVE_PRIOR_PATTERN.match(line)
        if match is not None:
            priors[match.group('move')] = float(match.group('prior')) / 100.0
        wdl_match = WDL_PATTERN.search(line)
        if wdl_match is not None:
            win, draw, loss = (int(value) / 1000.0 for value in wdl_match.groups())
            wdl = (win, draw, loss)
    if not priors:
        raise SystemExit(
            'Parsed no move priors from Lc0. Check the binary version and that --verbose-move-stats is '
            f'supported.\nCommand: {" ".join(command)}\nLast output:\n' + '\n'.join(output[-40:])
        )
    return TeacherReference(priors=priors, wdl=wdl)


def standard_uci(fen: str, move_uci: str) -> str:
    """Lc0 may print castling as king-takes-rook; compare in standard UCI."""
    import chess

    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if board.is_castling(move):
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        return move_uci[:2] + ('g' if kingside else 'c') + move_uci[1]
    return move_uci


def position_fen(moves_uci: tuple[str, ...]) -> str:
    position = CHESS_STATE_CONTRACT.initial_position()
    for move_uci in moves_uci:
        position = CHESS_STATE_CONTRACT.child_position(position, position.action_id_from_uci(move_uci))
    return position.fen


def query_wrapped_teacher(
    model: torch.jit.ScriptModule, device: torch.device, moves_uci: tuple[str, ...]
) -> tuple[dict[str, float], tuple[float, float, float]]:
    position = CHESS_STATE_CONTRACT.initial_position()
    for move_uci in moves_uci:
        position = CHESS_STATE_CONTRACT.child_position(position, position.action_id_from_uci(move_uci))
    legal_action_ids = np.asarray(CHESS_STATE_CONTRACT.legal_action_ids(position), dtype=np.int64)
    decoded = decode_lc0_planes((position.lc0_packed_encoding(),)).astype(np.float32)
    with torch.inference_mode():
        policy_logits, wdl = model(torch.from_numpy(decoded).to(device=device, dtype=teacher_input_dtype(model)))
    legal_logits = policy_logits[0].float().cpu().numpy()[legal_action_ids].astype(np.float64)
    shifted = np.exp(legal_logits - legal_logits.max())
    probabilities = shifted / shifted.sum()
    priors = {
        position.action_uci(int(action_id)): float(probability)
        for action_id, probability in zip(legal_action_ids, probabilities, strict=True)
    }
    wdl_row = wdl[0].float().cpu().numpy()
    return priors, (float(wdl_row[0]), float(wdl_row[1]), float(wdl_row[2]))


def legal_priors_batch(
    model: torch.jit.ScriptModule, device: torch.device, dtype: torch.dtype, positions: list[object]
) -> list[np.ndarray]:
    planes = decode_lc0_planes(tuple(position.lc0_packed_encoding() for position in positions)).astype(np.float32)
    with torch.inference_mode():
        policy_logits, _ = model(torch.from_numpy(planes).to(device=device, dtype=dtype))
    logits = policy_logits.float().cpu().numpy()
    priors: list[np.ndarray] = []
    for row, position in enumerate(positions):
        legal = np.asarray(CHESS_STATE_CONTRACT.legal_action_ids(position), dtype=np.int64)
        values = logits[row, legal].astype(np.float64)
        shifted = np.exp(values - values.max())
        priors.append(shifted / shifted.sum())
    return priors


def batch_and_precision_checks(
    teacher_model: Path, device: torch.device, opening_lines: tuple[tuple[str, ...], ...]
) -> tuple[float, float | None, float]:
    positions: list[object] = []
    for moves_uci in opening_lines:
        position = CHESS_STATE_CONTRACT.initial_position()
        for move_uci in moves_uci:
            position = CHESS_STATE_CONTRACT.child_position(position, position.action_id_from_uci(move_uci))
        positions.append(position)
    model = torch.jit.load(str(teacher_model), map_location=device).eval()
    batched = legal_priors_batch(model, device, teacher_input_dtype(model), positions)
    single = [legal_priors_batch(model, device, teacher_input_dtype(model), [position])[0] for position in positions]
    batch_difference = max(float(np.abs(a - b).max()) for a, b in zip(batched, single, strict=True))
    try:
        half = torch.jit.load(str(teacher_model), map_location=device).eval().to(torch.bfloat16)
        reduced = legal_priors_batch(half, device, torch.bfloat16, positions)
    except RuntimeError as error:
        print(f'bfloat16 evaluation raised: {error}')
        return batch_difference, None, 0.0
    bfloat16_difference = max(float(np.abs(a - b).max()) for a, b in zip(batched, reduced, strict=True))
    agreement = float(np.mean([int(np.argmax(a) == np.argmax(b)) for a, b in zip(batched, reduced, strict=True)]))
    return batch_difference, bfloat16_difference, agreement


def compare(
    reference: TeacherReference, ours: dict[str, float], our_wdl: tuple[float, float, float], moves_uci: tuple[str, ...]
) -> PositionComparison:
    missing = set(reference.priors) ^ set(ours)
    if missing:
        raise SystemExit(
            f'Legal move sets disagree after {" ".join(moves_uci)}: {sorted(missing)}. '
            'That is a move-mapping or legality bug, not a numerical one.'
        )
    maximum_difference = max(abs(reference.priors[move] - ours[move]) for move in reference.priors)
    reference_best = max(reference.priors, key=lambda move: reference.priors[move])
    our_best = max(ours, key=lambda move: ours[move])
    wdl_difference = (
        None if reference.wdl is None else max(abs(a - b) for a, b in zip(reference.wdl, our_wdl, strict=True))
    )
    return PositionComparison(
        moves_uci=moves_uci,
        maximum_policy_difference=maximum_difference,
        top_move_agrees=reference_best == our_best,
        wdl_difference=wdl_difference,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lc0-binary', type=Path, required=True)
    parser.add_argument('--lc0-network', type=Path, required=True)
    parser.add_argument('--teacher-model', type=Path, required=True, help='Output of build_lc0_teacher_model.py.')
    parser.add_argument('--openings', type=Path, required=True, help='Opening suite TSV supplying move sequences.')
    parser.add_argument('--positions', type=int, default=20)
    parser.add_argument('--nodes', type=int, default=1, help='Lc0 search nodes; 1 reports the raw root priors.')
    parser.add_argument('--policy-tolerance', type=float, default=2.0e-3)
    parser.add_argument('--wdl-tolerance', type=float, default=5.0e-3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--extra-lc0-argument', action='append', default=[])
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    device = torch.device(arguments.device)
    model = torch.jit.load(str(arguments.teacher_model), map_location=device).eval()
    opening_lines = read_opening_lines(arguments.openings, arguments.positions)

    comparisons: list[PositionComparison] = []
    for moves_uci in opening_lines:
        reference = query_lc0(
            arguments.lc0_binary, arguments.lc0_network, moves_uci, arguments.nodes, arguments.extra_lc0_argument
        )
        fen = position_fen(moves_uci)
        reference = TeacherReference(
            priors={standard_uci(fen, move): prior for move, prior in reference.priors.items()},
            wdl=reference.wdl,
        )
        ours, our_wdl = query_wrapped_teacher(model, device, moves_uci)
        comparisons.append(compare(reference, ours, our_wdl, moves_uci))

    worst_policy = max(comparison.maximum_policy_difference for comparison in comparisons)
    disagreements = [comparison for comparison in comparisons if not comparison.top_move_agrees]
    wdl_differences = [c.wdl_difference for c in comparisons if c.wdl_difference is not None]
    worst_wdl = max(wdl_differences) if wdl_differences else None

    print(f'Compared {len(comparisons)} positions against Lc0.')
    print(f'Worst policy difference: {worst_policy:.6f} (tolerance {arguments.policy_tolerance})')
    print(f'Top-move disagreements: {len(disagreements)}')
    if worst_wdl is None:
        print('WDL not reported by this Lc0 build; policy was compared alone.')
    else:
        print(f'Worst WDL difference: {worst_wdl:.6f} (tolerance {arguments.wdl_tolerance})')

    failed = worst_policy > arguments.policy_tolerance or bool(disagreements)
    if worst_wdl is not None and worst_wdl > arguments.wdl_tolerance:
        failed = True

    batch_difference, bfloat16_difference, bfloat16_top_agreement = batch_and_precision_checks(
        arguments.teacher_model, device, opening_lines
    )
    # Tracing an ONNX conversion can bake the sample batch size into reshapes; the search batches up to 64.
    print(f'Batched versus single-position policy difference: {batch_difference:.2e}')
    if batch_difference > 1.0e-4:
        failed = True
        print('  The wrapped teacher depends on batch composition; its trace is not batch-size generic.')
    if bfloat16_difference is None:
        print('BF16_UNUSABLE: the wrapped teacher fails in bfloat16, so the pipeline must serve it in float32.')
    else:
        print(
            f'bfloat16 versus float32: worst prior difference {bfloat16_difference:.4f}, '
            f'top-move agreement {bfloat16_top_agreement:.3f}'
        )
    for comparison in disagreements[:5]:
        print(f'  disagreed after: {" ".join(comparison.moves_uci)}')
    if failed:
        raise SystemExit('Teacher fidelity check FAILED. Do not run matches until the encoding agrees with Lc0.')
    print('Teacher fidelity check passed.')


if __name__ == '__main__':
    main()
