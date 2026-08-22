from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from src.evaluation.contracts import CandidateOutcome, EvaluationGameResult
from src.evaluation.ladder import (
    STOCKFISH_FIXED_NODES_ANCHOR_ELO,
    LadderRungObservation,
    LadderRungPairScores,
    bootstrap_ladder_elo_interval,
    fit_ladder_elo,
)
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from tools.run_stockfish_gauntlet import (
    Arguments as GauntletArguments,
    SeededOpeningSelection,
    StockfishGauntletResult,
    run_gauntlet,
)
from tools.search_budget import (
    FixedModelSearchBudget,
    ModelSearchBudget,
    TimedModelSearchBudget,
    add_model_search_budget_arguments,
    model_search_budget,
)
from src.util.hashing import file_sha256


class LadderProbe(FrozenModel):
    stockfish_nodes: int = Field(gt=0)
    result_path: Path
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    score_confidence_low: float = Field(ge=0.0, le=1.0)
    score_confidence_high: float = Field(ge=0.0, le=1.0)


class LadderBracket(FrozenModel):
    lower_stockfish_nodes: int = Field(gt=0)
    upper_stockfish_nodes: int = Field(gt=0)


class LadderEloFit(FrozenModel):
    anchor_source: Literal['melonimarco-ssdf-20260820'] = 'melonimarco-ssdf-20260820'
    bootstrap_samples: int = Field(gt=0)
    ladder_elo: float
    confidence_low: float
    confidence_high: float


class StockfishLadderResult(FrozenModel):
    schema_version: Literal[2] = 2
    source_revision: str = Field(min_length=40, max_length=40)
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    checkpoint_generation: int = Field(ge=0)
    opening_manifest_path: Path
    opening_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stockfish_executable_path: Path
    stockfish_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stockfish_identity: str = Field(min_length=1)
    probe_game_count: int = Field(gt=0)
    opening_pair_count: int = Field(gt=0)
    opening_selection_seed: int = Field(ge=0)
    match_random_seed: int = Field(ge=0)
    model_search_budget: ModelSearchBudget
    probes: tuple[LadderProbe, ...] = Field(min_length=1)
    closest_stockfish_nodes: int = Field(gt=0)
    score_bracket: LadderBracket | None
    ladder_elo_fit: LadderEloFit | None


@dataclass(frozen=True)
class Arguments:
    experiment: Path
    run_directory: Path
    checkpoint_generation: int
    opening_manifest: Path
    stockfish_executable: Path
    stockfish_node_ladder: tuple[int, ...]
    probe_games: int
    opening_selection_seed: int
    match_random_seed: int
    devices: tuple[int, ...]
    model_search_budget: FixedModelSearchBudget | TimedModelSearchBudget
    output_directory: Path


_LADDER_BOOTSTRAP_SAMPLES = 20_000
_OUTCOME_SCORE = {
    CandidateOutcome.WIN: 1.0,
    CandidateOutcome.DRAW: 0.5,
    CandidateOutcome.LOSS: 0.0,
}


def _pair_scores(games: tuple[EvaluationGameResult, ...]) -> tuple[float, ...]:
    scores_by_pair: dict[int, list[float]] = {}
    for game in games:
        scores_by_pair.setdefault(game.pair_index, []).append(_OUTCOME_SCORE[game.outcome])
    if any(len(scores) != 2 for scores in scores_by_pair.values()):
        raise ValueError('Every ladder opening pair must contain exactly two color-swapped games.')
    return tuple(sum(scores_by_pair[pair_index]) / 2.0 for pair_index in sorted(scores_by_pair))


def _ladder_elo_fit(
    pair_scores_by_nodes: dict[int, tuple[float, ...]],
    bootstrap_seed: int,
) -> LadderEloFit | None:
    if any(nodes not in STOCKFISH_FIXED_NODES_ANCHOR_ELO for nodes in pair_scores_by_nodes):
        return None
    observations = tuple(
        LadderRungObservation(
            anchor_elo=STOCKFISH_FIXED_NODES_ANCHOR_ELO[nodes],
            score=sum(pair_scores) / len(pair_scores),
            game_count=2 * len(pair_scores),
        )
        for nodes, pair_scores in sorted(pair_scores_by_nodes.items())
    )
    rungs = tuple(
        LadderRungPairScores(
            anchor_elo=STOCKFISH_FIXED_NODES_ANCHOR_ELO[nodes],
            pair_scores=pair_scores,
        )
        for nodes, pair_scores in sorted(pair_scores_by_nodes.items())
    )
    confidence_low, confidence_high = bootstrap_ladder_elo_interval(
        rungs,
        _LADDER_BOOTSTRAP_SAMPLES,
        bootstrap_seed,
    )
    return LadderEloFit(
        bootstrap_samples=_LADDER_BOOTSTRAP_SAMPLES,
        ladder_elo=fit_ladder_elo(observations),
        confidence_low=confidence_low,
        confidence_high=confidence_high,
    )


def _score_bracket(probes: tuple[LadderProbe, ...]) -> LadderBracket | None:
    ordered = tuple(sorted(probes, key=lambda probe: probe.stockfish_nodes))
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        if lower.score >= 0.5 and upper.score <= 0.5:
            return LadderBracket(
                lower_stockfish_nodes=lower.stockfish_nodes,
                upper_stockfish_nodes=upper.stockfish_nodes,
            )
    return None


def run_ladder(arguments: Arguments) -> StockfishLadderResult:
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    opening_pairs = arguments.probe_games // 2
    probes: list[LadderProbe] = []
    pair_scores_by_nodes: dict[int, tuple[float, ...]] = {}
    first_gauntlet_result: StockfishGauntletResult | None = None
    for stockfish_nodes in arguments.stockfish_node_ladder:
        run_directory = arguments.output_directory / f'stockfish-nodes-{stockfish_nodes}'
        result = run_gauntlet(
            GauntletArguments(
                experiment=arguments.experiment,
                run_directory=arguments.run_directory,
                checkpoint_generation=arguments.checkpoint_generation,
                opening_manifest=arguments.opening_manifest,
                stockfish_executable=arguments.stockfish_executable,
                stockfish_nodes=stockfish_nodes,
                opening_pairs=opening_pairs,
                opening_selection=SeededOpeningSelection(random_seed=arguments.opening_selection_seed),
                match_random_seed=arguments.match_random_seed,
                devices=arguments.devices,
                model_search_budget=arguments.model_search_budget,
                output_directory=run_directory,
            )
        )
        aggregate = result.aggregate
        pair_scores_by_nodes[stockfish_nodes] = _pair_scores(result.games)
        if first_gauntlet_result is None:
            first_gauntlet_result = result
        probes.append(
            LadderProbe(
                stockfish_nodes=stockfish_nodes,
                result_path=(run_directory / 'result.json').resolve(),
                wins=aggregate.wins,
                draws=aggregate.draws,
                losses=aggregate.losses,
                score=aggregate.score,
                score_confidence_low=aggregate.score_confidence_low,
                score_confidence_high=aggregate.score_confidence_high,
            )
        )
    ordered_probes = tuple(sorted(probes, key=lambda probe: probe.stockfish_nodes))
    assert first_gauntlet_result is not None
    result = StockfishLadderResult(
        source_revision=first_gauntlet_result.source_revision,
        tool_sha256=file_sha256(Path(__file__)),
        checkpoint_generation=arguments.checkpoint_generation,
        opening_manifest_path=arguments.opening_manifest.resolve(),
        opening_manifest_sha256=first_gauntlet_result.opening_manifest_sha256,
        stockfish_executable_path=arguments.stockfish_executable.resolve(),
        stockfish_executable_sha256=first_gauntlet_result.stockfish_executable_sha256,
        stockfish_identity=first_gauntlet_result.stockfish_identity,
        probe_game_count=arguments.probe_games,
        opening_pair_count=opening_pairs,
        opening_selection_seed=arguments.opening_selection_seed,
        match_random_seed=arguments.match_random_seed,
        model_search_budget=arguments.model_search_budget,
        probes=ordered_probes,
        closest_stockfish_nodes=min(
            ordered_probes,
            key=lambda probe: (abs(probe.score - 0.5), probe.stockfish_nodes),
        ).stockfish_nodes,
        score_bracket=_score_bracket(ordered_probes),
        ladder_elo_fit=_ladder_elo_fit(pair_scores_by_nodes, arguments.match_random_seed),
    )
    write_text_atomically(arguments.output_directory / 'ladder-result.json', result.model_dump_json(indent=2) + '\n')
    return result


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Probe several Stockfish node rungs with one paired opening sample.')
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--opening-manifest', required=True, type=Path)
    parser.add_argument('--stockfish-executable', required=True, type=Path)
    parser.add_argument('--stockfish-node-ladder', required=True, nargs='+', type=int)
    parser.add_argument('--probe-games', choices=(10, 20), default=20, type=int)
    parser.add_argument('--opening-selection-seed', default=20260815, type=int)
    parser.add_argument('--match-random-seed', default=20260816, type=int)
    parser.add_argument('--devices', required=True, nargs='+', type=int)
    add_model_search_budget_arguments(parser)
    parser.add_argument('--output-directory', required=True, type=Path)
    namespace = parser.parse_args()
    arguments = Arguments(
        experiment=namespace.experiment,
        run_directory=namespace.run_directory,
        checkpoint_generation=namespace.checkpoint_generation,
        opening_manifest=namespace.opening_manifest,
        stockfish_executable=namespace.stockfish_executable,
        stockfish_node_ladder=tuple(namespace.stockfish_node_ladder),
        probe_games=namespace.probe_games,
        opening_selection_seed=namespace.opening_selection_seed,
        match_random_seed=namespace.match_random_seed,
        devices=tuple(namespace.devices),
        model_search_budget=model_search_budget(namespace),
        output_directory=namespace.output_directory,
    )
    required_paths = (
        arguments.experiment,
        arguments.run_directory,
        arguments.opening_manifest,
        arguments.stockfish_executable,
    )
    if not all(path.exists() for path in required_paths):
        raise ValueError('Experiment, run directory, openings, and Stockfish executable must exist.')
    if (
        arguments.checkpoint_generation < 0
        or arguments.match_random_seed < 0
        or any(nodes <= 0 for nodes in arguments.stockfish_node_ladder)
        or any(device < 0 for device in arguments.devices)
    ):
        raise ValueError('Checkpoint generation, node counts, and device IDs are invalid.')
    if len(set(arguments.stockfish_node_ladder)) != len(arguments.stockfish_node_ladder):
        raise ValueError('Stockfish ladder node counts must be unique.')
    if not arguments.devices or len(set(arguments.devices)) != len(arguments.devices):
        raise ValueError('Ladder devices must be nonempty and unique.')
    if arguments.output_directory.exists():
        raise ValueError(f'Ladder output directory already exists: {arguments.output_directory}')
    return arguments


def main() -> None:
    print(run_ladder(parse_arguments()).model_dump_json(indent=2))


if __name__ == '__main__':
    main()
