"""Generates distillation data by playing games with an Lc0 evaluator and no search at all.

One network evaluation per position: the position is evaluated, the policy is masked to legal moves,
scaled by temperature and sampled from, and the move is played. The stored target is the teacher's
plain policy and WDL for that exact position, with no MCTS visit counts, no terminal outcome and no
discounting, so the label is the teacher's function rather than anything this project derived.

Sampling matches production self-play in shape: up to eight random opening plies, temperature
interpolated from `starting-temperature` to `final-temperature` across `greedy-after-ply`, argmax
after it. Production applies that temperature to visit counts, which do not exist here, so it is
applied to the network policy instead.

Requires the native extension, so it runs on the node.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from src.games.chess.contract import ChessStateContract
from src.games.representation import PackedPlanePayload, decode_packed_planes_batch

CHESS_STATE = ChessStateContract()


@dataclass(frozen=True)
class SamplingParameters:
    starting_temperature: float
    final_temperature: float
    greedy_after_ply: int
    maximum_random_opening_plies: int
    maximum_game_plies: int


@dataclass
class ActiveGame:
    position: object
    ply: int
    records: list[dict[str, object]]


def unpack_planes(payloads: list[bytes]) -> torch.Tensor:
    """Reuses the replay decoder so the teacher sees exactly the bytes the native encoder produced."""
    states = tuple(PackedPlanePayload(payload) for payload in payloads)
    planes = decode_packed_planes_batch(
        states,
        CHESS_STATE.packed_plane_layout,
        CHESS_STATE.representation.binary_channels,
        CHESS_STATE.representation.scalar_channels,
    )
    return torch.from_numpy(planes)


def temperature_at(ply: int, parameters: SamplingParameters) -> float:
    progress = min(ply / parameters.greedy_after_ply, 1.0)
    return parameters.starting_temperature + (parameters.final_temperature - parameters.starting_temperature) * progress


def select_action(
    legal_action_ids: list[int],
    legal_policy: np.ndarray,
    ply: int,
    parameters: SamplingParameters,
    generator: random.Random,
) -> int:
    if ply >= parameters.greedy_after_ply:
        return legal_action_ids[int(np.argmax(legal_policy))]
    weights = np.power(legal_policy, 1.0 / temperature_at(ply, parameters))
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return legal_action_ids[int(np.argmax(legal_policy))]
    weights = weights / total
    return legal_action_ids[int(np.asarray(generator.choices(range(len(weights)), weights=weights, k=1))[0])]


def new_game(parameters: SamplingParameters, generator: random.Random) -> ActiveGame:
    import AlphaZeroCpp

    position = AlphaZeroCpp.ChessPosition()
    for _ in range(generator.randint(0, parameters.maximum_random_opening_plies)):
        if position.is_terminal:
            break
        position = position.child(generator.choice(position.legal_actions()))
    return ActiveGame(position=position, ply=0, records=[])


def evaluate(
    model: torch.jit.ScriptModule, games: list[ActiveGame], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    planes = unpack_planes([game.position.packed_encoding() for game in games]).to(device)
    with torch.inference_mode():
        policy_logits, wdl = model(planes)
    return policy_logits.float().cpu().numpy(), wdl.float().cpu().numpy()


def run(
    model: torch.jit.ScriptModule,
    device: torch.device,
    parameters: SamplingParameters,
    target_positions: int,
    concurrent_games: int,
    seed: int,
    output: Path,
) -> None:
    generator = random.Random(seed)
    games = [new_game(parameters, generator) for _ in range(concurrent_games)]
    written = 0
    completed_games = 0
    started_at = time.time()
    with output.open('w', encoding='utf-8') as sink:
        while written < target_positions:
            policy_logits, wdl = evaluate(model, games, device)
            for index, game in enumerate(games):
                legal_action_ids = game.position.legal_actions()
                logits = policy_logits[index][legal_action_ids]
                shifted = np.exp(logits - logits.max())
                legal_policy = shifted / shifted.sum()
                game.records.append(
                    {
                        'packed_state': game.position.packed_encoding().hex(),
                        'legal_action_ids': legal_action_ids,
                        'policy': [round(float(value), 6) for value in legal_policy],
                        'wdl': [round(float(value), 6) for value in wdl[index]],
                        'ply': game.ply,
                    }
                )
                action_id = select_action(legal_action_ids, legal_policy, game.ply, parameters, generator)
                game.position = game.position.child(action_id)
                game.ply += 1
                if game.position.is_terminal or game.ply >= parameters.maximum_game_plies:
                    for record in game.records:
                        sink.write(json.dumps(record) + '\n')
                    written += len(game.records)
                    completed_games += 1
                    games[index] = new_game(parameters, generator)
            if completed_games and completed_games % 50 == 0:
                rate = written / max(time.time() - started_at, 1e-6)
                print(f'{written} positions from {completed_games} games, {rate:.0f} positions/s')
    print(f'Wrote {written} positions from {completed_games} games to {output}')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model', type=Path, required=True, help='TorchScript teacher from build_lc0_teacher_model.py.'
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-positions', type=int, required=True)
    parser.add_argument('--concurrent-games', type=int, default=256, help='Batch width for teacher evaluation.')
    parser.add_argument('--starting-temperature', type=float, default=1.3)
    parser.add_argument('--final-temperature', type=float, default=0.1)
    parser.add_argument('--greedy-after-ply', type=int, default=80)
    parser.add_argument('--maximum-random-opening-plies', type=int, default=8)
    parser.add_argument('--maximum-game-plies', type=int, default=300)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260926)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    device = torch.device(arguments.device)
    model = torch.jit.load(str(arguments.model), map_location=device).eval()
    parameters = SamplingParameters(
        starting_temperature=arguments.starting_temperature,
        final_temperature=arguments.final_temperature,
        greedy_after_ply=arguments.greedy_after_ply,
        maximum_random_opening_plies=arguments.maximum_random_opening_plies,
        maximum_game_plies=arguments.maximum_game_plies,
    )
    run(
        model=model,
        device=device,
        parameters=parameters,
        target_positions=arguments.target_positions,
        concurrent_games=arguments.concurrent_games,
        seed=arguments.seed,
        output=arguments.output,
    )


if __name__ == '__main__':
    main()
