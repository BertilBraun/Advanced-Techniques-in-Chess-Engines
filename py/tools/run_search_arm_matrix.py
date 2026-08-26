from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, TypeAdapter, model_validator
from src.evaluation.contracts import CandidateOutcome, EvaluationGameResult
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision

if TYPE_CHECKING:
    from tools.run_stockfish_gauntlet import StockfishGauntletResult

_BOOTSTRAP_SAMPLES = 20_000
_OUTCOME_SCORE = {
    CandidateOutcome.WIN: 1.0,
    CandidateOutcome.DRAW: 0.5,
    CandidateOutcome.LOSS: 0.0,
}


class SearchArm(FrozenModel):
    label: str = Field(min_length=1, pattern=r'^[a-z0-9][a-z0-9-]*$')
    searches_per_move: int = Field(gt=0)
    parallel_searches: int = Field(gt=0)
    exploration_constant: float = Field(gt=0.0)
    first_play_urgency: Literal['zero', 'parent_value', 'reduced_parent_value']
    first_play_urgency_reduction: float | None = Field(default=None, gt=0.0)
    virtual_loss_weight: float = Field(ge=0.0, le=1.0)
    search_value_discount_per_ply: float = Field(gt=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_reduction(self) -> SearchArm:
        if (self.first_play_urgency == 'reduced_parent_value') != (self.first_play_urgency_reduction is not None):
            raise ValueError('A first-play-urgency reduction belongs to reduced_parent_value and nowhere else.')
        if self.searches_per_move <= self.parallel_searches:
            raise ValueError('Arm searches per move must exceed its parallel searches.')
        return self

    def gauntlet_arguments(self) -> tuple[str, ...]:
        reduction = (
            ()
            if self.first_play_urgency_reduction is None
            else ('--first-play-urgency-reduction', str(self.first_play_urgency_reduction))
        )
        return (
            '--model-searches',
            str(self.searches_per_move),
            '--parallel-searches',
            str(self.parallel_searches),
            '--exploration-constant',
            str(self.exploration_constant),
            '--first-play-urgency',
            self.first_play_urgency,
            *reduction,
            '--virtual-loss-weight',
            str(self.virtual_loss_weight),
            '--search-value-discount-per-ply',
            str(self.search_value_discount_per_ply),
        )


class SearchArmMatrix(FrozenModel):
    schema_version: Literal[1] = 1
    description: str = Field(min_length=1)
    baseline_label: str = Field(min_length=1)
    arms: tuple[SearchArm, ...] = Field(min_length=2)

    @model_validator(mode='after')
    def validate_labels(self) -> SearchArmMatrix:
        labels = tuple(arm.label for arm in self.arms)
        if len(set(labels)) != len(labels):
            raise ValueError('Search arm labels must be unique.')
        if self.baseline_label not in labels:
            raise ValueError(f'Baseline arm {self.baseline_label!r} is not present in the matrix.')
        return self


SEARCH_ARM_MATRIX_ADAPTER = TypeAdapter(SearchArmMatrix)


@dataclass(frozen=True)
class Arguments:
    matrix: Path
    experiment: Path
    run_directory: Path
    checkpoint_generation: int
    opening_manifest: Path
    stockfish_executable: Path
    stockfish_nodes: int
    opening_pairs: int
    opening_selection_seed: int
    match_random_seed: int
    device: int
    concurrency: int
    inference_batch_size: int
    output_directory: Path


class ArmOutcome(FrozenModel):
    label: str = Field(min_length=1)
    arm: SearchArm
    result_path: Path
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    score_confidence_low: float = Field(ge=0.0, le=1.0)
    score_confidence_high: float = Field(ge=0.0, le=1.0)
    duration_seconds: float = Field(ge=0.0)


class PairedDifference(FrozenModel):
    label: str = Field(min_length=1)
    score_difference: float = Field(ge=-1.0, le=1.0)
    difference_confidence_low: float
    difference_confidence_high: float
    excludes_zero: bool


class SearchArmMatrixResult(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    matrix_path: Path
    matrix_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    experiment_path: Path
    run_directory: Path
    checkpoint_generation: int = Field(ge=0)
    opening_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stockfish_identity: str = Field(min_length=1)
    stockfish_nodes: int = Field(gt=0)
    opening_pairs: int = Field(gt=0)
    opening_selection_seed: int = Field(ge=0)
    match_random_seed: int = Field(ge=0)
    device: int = Field(ge=0)
    concurrency: int = Field(gt=0)
    bootstrap_samples: int = Field(gt=0)
    baseline_label: str = Field(min_length=1)
    arms: tuple[ArmOutcome, ...] = Field(min_length=1)
    paired_differences: tuple[PairedDifference, ...]
    duration_seconds: float = Field(ge=0.0)


def _pair_scores(games: tuple[EvaluationGameResult, ...]) -> tuple[float, ...]:
    scores_by_pair: dict[int, list[float]] = {}
    for game in games:
        scores_by_pair.setdefault(game.pair_index, []).append(_OUTCOME_SCORE[game.outcome])
    if any(len(scores) != 2 for scores in scores_by_pair.values()):
        raise ValueError('Every arm opening pair must contain exactly two colour-swapped games.')
    return tuple(sum(scores_by_pair[pair_index]) / 2.0 for pair_index in sorted(scores_by_pair))


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_difference(
    label: str,
    arm_scores: tuple[float, ...],
    baseline_scores: tuple[float, ...],
    random_seed: int,
) -> PairedDifference:
    if len(arm_scores) != len(baseline_scores):
        raise ValueError('Paired arms must cover the same opening pairs.')
    # Both arms played the same openings with the same colours, so the per-pair difference removes
    # opening difficulty from the comparison and is the quantity worth bootstrapping.
    differences = [arm - baseline for arm, baseline in zip(arm_scores, baseline_scores, strict=True)]
    generator = random.Random(random_seed)
    samples = sorted(
        sum(differences[generator.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(_BOOTSTRAP_SAMPLES)
    )
    low = _quantile(samples, 0.025)
    high = _quantile(samples, 0.975)
    return PairedDifference(
        label=label,
        score_difference=sum(differences) / len(differences),
        difference_confidence_low=low,
        difference_confidence_high=high,
        excludes_zero=low > 0.0 or high < 0.0,
    )


def _arm_command(arguments: Arguments, arm: SearchArm) -> tuple[str, ...]:
    return (
        sys.executable,
        '-m',
        'tools.run_stockfish_gauntlet',
        '--experiment',
        str(arguments.experiment),
        '--run-directory',
        str(arguments.run_directory),
        '--checkpoint-generation',
        str(arguments.checkpoint_generation),
        '--opening-manifest',
        str(arguments.opening_manifest),
        '--stockfish-executable',
        str(arguments.stockfish_executable),
        '--stockfish-nodes',
        str(arguments.stockfish_nodes),
        '--opening-pairs',
        str(arguments.opening_pairs),
        '--opening-selection',
        'seeded_sample',
        '--opening-selection-seed',
        str(arguments.opening_selection_seed),
        '--match-random-seed',
        str(arguments.match_random_seed),
        '--devices',
        str(arguments.device),
        '--inference-batch-size',
        str(arguments.inference_batch_size),
        '--output-directory',
        str(arguments.output_directory / arm.label),
        *arm.gauntlet_arguments(),
    )


def _run_arm(arguments: Arguments, arm: SearchArm) -> tuple[SearchArm, StockfishGauntletResult, float]:
    from tools.run_stockfish_gauntlet import StockfishGauntletResult

    started = time.monotonic()
    log_path = arguments.output_directory / f'{arm.label}.log'
    with log_path.open('w', encoding='utf-8') as log:
        completed = subprocess.run(
            _arm_command(arguments, arm),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f'Arm {arm.label!r} failed with exit status {completed.returncode}; see {log_path}.')
    result_path = arguments.output_directory / arm.label / 'result.json'
    result = StockfishGauntletResult.model_validate_json(result_path.read_text(encoding='utf-8'))
    return arm, result, time.monotonic() - started


def run_matrix(arguments: Arguments) -> SearchArmMatrixResult:
    matrix = SEARCH_ARM_MATRIX_ADAPTER.validate_json(arguments.matrix.read_text(encoding='utf-8'))
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed: dict[str, tuple[SearchArm, StockfishGauntletResult, float]] = {}
    with ThreadPoolExecutor(max_workers=arguments.concurrency) as executor:
        futures = {executor.submit(_run_arm, arguments, arm): arm.label for arm in matrix.arms}
        for future in as_completed(futures):
            arm, result, duration = future.result()
            completed[arm.label] = (arm, result, duration)
            print(f'{arm.label}: score {result.aggregate.score:.3f} in {duration:.0f} s', flush=True)

    identities = {result.stockfish_identity for _, result, _ in completed.values()}
    if len(identities) != 1:
        raise ValueError(f'Arm Stockfish identities disagree: {sorted(identities)}')
    manifest_hashes = {result.opening_manifest_sha256 for _, result, _ in completed.values()}
    if len(manifest_hashes) != 1:
        raise ValueError('Arms did not share one opening manifest.')

    ordered = tuple(completed[arm.label] for arm in matrix.arms)
    baseline_scores = _pair_scores(completed[matrix.baseline_label][1].games)
    return SearchArmMatrixResult(
        source_revision=read_source_revision().commit,
        tool_sha256=file_sha256(Path(__file__)),
        matrix_path=arguments.matrix.resolve(),
        matrix_sha256=file_sha256(arguments.matrix),
        experiment_path=arguments.experiment.resolve(),
        run_directory=arguments.run_directory.resolve(),
        checkpoint_generation=arguments.checkpoint_generation,
        opening_manifest_sha256=next(iter(manifest_hashes)),
        stockfish_identity=next(iter(identities)),
        stockfish_nodes=arguments.stockfish_nodes,
        opening_pairs=arguments.opening_pairs,
        opening_selection_seed=arguments.opening_selection_seed,
        match_random_seed=arguments.match_random_seed,
        device=arguments.device,
        concurrency=arguments.concurrency,
        bootstrap_samples=_BOOTSTRAP_SAMPLES,
        baseline_label=matrix.baseline_label,
        arms=tuple(
            ArmOutcome(
                label=arm.label,
                arm=arm,
                result_path=(arguments.output_directory / arm.label / 'result.json').resolve(),
                wins=result.aggregate.wins,
                draws=result.aggregate.draws,
                losses=result.aggregate.losses,
                score=result.aggregate.score,
                score_confidence_low=result.aggregate.score_confidence_low,
                score_confidence_high=result.aggregate.score_confidence_high,
                duration_seconds=duration,
            )
            for arm, result, duration in ordered
        ),
        paired_differences=tuple(
            paired_difference(
                arm.label,
                _pair_scores(result.games),
                baseline_scores,
                arguments.match_random_seed,
            )
            for arm, result, _ in ordered
            if arm.label != matrix.baseline_label
        ),
        duration_seconds=time.monotonic() - started,
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run one Stockfish rung across a matrix of model search arms.')
    parser.add_argument('--matrix', required=True, type=Path)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--opening-manifest', required=True, type=Path)
    parser.add_argument('--stockfish-executable', required=True, type=Path)
    parser.add_argument('--stockfish-nodes', required=True, type=int)
    parser.add_argument('--opening-pairs', required=True, type=int)
    parser.add_argument('--opening-selection-seed', default=20260826, type=int)
    parser.add_argument('--match-random-seed', default=20260827, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--concurrency', default=4, type=int)
    parser.add_argument('--inference-batch-size', default=64, type=int)
    parser.add_argument('--output-directory', required=True, type=Path)
    parsed = parser.parse_args()
    arguments = Arguments(
        matrix=parsed.matrix,
        experiment=parsed.experiment,
        run_directory=parsed.run_directory,
        checkpoint_generation=parsed.checkpoint_generation,
        opening_manifest=parsed.opening_manifest,
        stockfish_executable=parsed.stockfish_executable,
        stockfish_nodes=parsed.stockfish_nodes,
        opening_pairs=parsed.opening_pairs,
        opening_selection_seed=parsed.opening_selection_seed,
        match_random_seed=parsed.match_random_seed,
        device=parsed.device,
        concurrency=parsed.concurrency,
        inference_batch_size=parsed.inference_batch_size,
        output_directory=parsed.output_directory,
    )
    required_paths = (
        arguments.matrix,
        arguments.experiment,
        arguments.run_directory,
        arguments.opening_manifest,
        arguments.stockfish_executable,
    )
    if not all(path.exists() for path in required_paths):
        raise ValueError('Matrix, experiment, run directory, openings and Stockfish executable must exist.')
    if min(arguments.stockfish_nodes, arguments.opening_pairs, arguments.concurrency) <= 0:
        raise ValueError('Stockfish nodes, opening pairs and concurrency must be positive.')
    if min(arguments.checkpoint_generation, arguments.device, arguments.match_random_seed) < 0:
        raise ValueError('Checkpoint generation, device and seeds must be nonnegative.')
    if arguments.output_directory.exists():
        raise ValueError(f'Arm matrix output directory already exists: {arguments.output_directory}')
    return arguments


def main() -> None:
    arguments = parse_arguments()
    result = run_matrix(arguments)
    write_text_atomically(
        arguments.output_directory / 'arm-matrix-result.json',
        result.model_dump_json(indent=2) + '\n',
    )
    print(arguments.output_directory / 'arm-matrix-result.json')


if __name__ == '__main__':
    main()
