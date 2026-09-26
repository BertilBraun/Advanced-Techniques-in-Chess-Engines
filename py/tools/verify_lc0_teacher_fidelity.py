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
from src.evaluation.inference import decode_packed_inputs
from src.games.chess.contract import CHESS_STATE_CONTRACT

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
        '--threads=1',
        '--minibatch-size=1',
        *extra_arguments,
    ]
    script = '\n'.join(
        [
            'uci',
            'setoption name UCI_ShowWDL value true',
            'isready',
            f'position startpos moves {" ".join(moves_uci)}',
            f'go nodes {nodes}',
            'quit',
            '',
        ]
    )
    completed = subprocess.run(command, input=script, capture_output=True, text=True, timeout=180, check=False)
    priors: dict[str, float] = {}
    wdl: tuple[float, float, float] | None = None
    for line in completed.stdout.splitlines():
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
            f'supported.\nCommand: {" ".join(command)}\nLast output:\n{completed.stdout[-2000:]}'
        )
    return TeacherReference(priors=priors, wdl=wdl)


def query_wrapped_teacher(
    model: torch.jit.ScriptModule, device: torch.device, moves_uci: tuple[str, ...]
) -> tuple[dict[str, float], tuple[float, float, float]]:
    position = CHESS_STATE_CONTRACT.initial_position()
    for move_uci in moves_uci:
        position = CHESS_STATE_CONTRACT.child_position(position, position.action_id_from_uci(move_uci))
    legal_action_ids = np.asarray(CHESS_STATE_CONTRACT.legal_action_ids(position), dtype=np.int64)
    decoded = decode_packed_inputs(CHESS_STATE_CONTRACT, (CHESS_STATE_CONTRACT.encode_network_input(position),))
    with torch.inference_mode():
        policy_logits, wdl = model(torch.from_numpy(decoded).to(device))
    legal_logits = policy_logits[0].float().cpu().numpy()[legal_action_ids].astype(np.float64)
    shifted = np.exp(legal_logits - legal_logits.max())
    probabilities = shifted / shifted.sum()
    priors = {
        position.action_uci(int(action_id)): float(probability)
        for action_id, probability in zip(legal_action_ids, probabilities, strict=True)
    }
    wdl_row = wdl[0].float().cpu().numpy()
    return priors, (float(wdl_row[0]), float(wdl_row[1]), float(wdl_row[2]))


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
    for comparison in disagreements[:5]:
        print(f'  disagreed after: {" ".join(comparison.moves_uci)}')
    if failed:
        raise SystemExit('Teacher fidelity check FAILED. Do not run matches until the encoding agrees with Lc0.')
    print('Teacher fidelity check passed.')


if __name__ == '__main__':
    main()
