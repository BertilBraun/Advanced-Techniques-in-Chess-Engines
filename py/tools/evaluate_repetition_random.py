from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.eval.ModelEvaluationCpp import ModelEvaluation
from src.experiment.run_configuration import apply_run_configuration, load_run_configuration
from src.settings import TRAINING_ARGS


@dataclass(frozen=True)
class Arguments:
    run_configuration: Path
    save_path: Path
    opening_suite: Path
    iteration: int
    device: int
    games: int
    searches: int
    seed: int


@dataclass(frozen=True)
class EvaluationResult:
    iteration: int
    games: int
    searches_per_turn: int
    seed: int
    wins: int
    draws: int
    losses: int


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-configuration', required=True, type=Path)
    parser.add_argument('--save-path', required=True, type=Path)
    parser.add_argument('--opening-suite', required=True, type=Path)
    parser.add_argument('--iteration', required=True, type=int)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--searches', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260719)
    namespace = parser.parse_args()
    return Arguments(
        run_configuration=namespace.run_configuration,
        save_path=namespace.save_path,
        opening_suite=namespace.opening_suite,
        iteration=namespace.iteration,
        device=namespace.device,
        games=namespace.games,
        searches=namespace.searches,
        seed=namespace.seed,
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.games < 2 or arguments.games % 2 != 0:
        raise ValueError('games must be a positive even number.')
    if arguments.searches < 1:
        raise ValueError('searches must be positive.')

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    configuration = load_run_configuration(arguments.run_configuration)
    apply_run_configuration(TRAINING_ARGS, configuration)
    TRAINING_ARGS.save_path = str(arguments.save_path.resolve())
    assert TRAINING_ARGS.evaluation is not None
    TRAINING_ARGS.evaluation.opening_suite_path = str(arguments.opening_suite.resolve())

    evaluator = ModelEvaluation(
        iteration=arguments.iteration,
        args=TRAINING_ARGS,
        device_id=arguments.device,
        num_games=arguments.games,
        num_searches_per_turn=arguments.searches,
    )
    results = evaluator.play_vs_random()
    result = EvaluationResult(
        iteration=arguments.iteration,
        games=arguments.games,
        searches_per_turn=arguments.searches,
        seed=arguments.seed,
        wins=results.wins,
        draws=results.draws,
        losses=results.losses,
    )
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == '__main__':
    main()
