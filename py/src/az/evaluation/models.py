from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from src.az.config.base import FrozenModel, Sha256
from src.az.config.seeds import (
    EvaluationActionSeedCoordinates,
    EvaluationGameSeedCoordinates,
    EvaluationSearchSeedCoordinates,
    SeedDerivationVersion,
    SeedPurpose,
    derive_seed,
)
from src.az.replay.envelope import GameTermination


class GoColor(str, Enum):
    BLACK = 'black'
    WHITE = 'white'


class EvaluationCostCategory(str, Enum):
    EVALUATION = 'evaluation'


class CandidateCheckpointIdentity(FrozenModel):
    checkpoint_id: UUID
    model_artifact_sha256: Sha256
    model_version: int = Field(gt=0)


class RandomOpponentIdentity(FrozenModel):
    kind: Literal['random']


class CheckpointOpponentIdentity(FrozenModel):
    kind: Literal['checkpoint']
    checkpoint: CandidateCheckpointIdentity


EvaluationOpponentIdentity = Annotated[
    RandomOpponentIdentity | CheckpointOpponentIdentity,
    Field(discriminator='kind'),
]


class EvaluationSeedLineage(FrozenModel):
    derivation_version: SeedDerivationVersion
    root_seed: int = Field(ge=0, le=2**63 - 1)
    evaluation_index: int = Field(ge=0)
    pair_index: int = Field(ge=0)
    game_in_pair: int = Field(ge=0, le=1)
    game_seed: int = Field(ge=0, le=2**63 - 1)
    search_seeds: tuple[int, ...]
    action_seeds: tuple[int, ...]

    @model_validator(mode='after')
    def validate_ply_streams(self) -> EvaluationSeedLineage:
        if len(self.search_seeds) != len(self.action_seeds):
            raise ValueError('Evaluation search and action seed streams must have equal length.')
        expected_game = derive_seed(
            self.root_seed,
            EvaluationGameSeedCoordinates(
                purpose=SeedPurpose.EVALUATION_GAME,
                evaluation_index=self.evaluation_index,
                pair_index=self.pair_index,
                game_in_pair=self.game_in_pair,
            ),
        )
        expected_search = tuple(
            derive_seed(
                self.root_seed,
                EvaluationSearchSeedCoordinates(
                    purpose=SeedPurpose.EVALUATION_SEARCH,
                    evaluation_index=self.evaluation_index,
                    pair_index=self.pair_index,
                    game_in_pair=self.game_in_pair,
                    ply=ply,
                ),
            )
            for ply in range(len(self.search_seeds))
        )
        expected_action = tuple(
            derive_seed(
                self.root_seed,
                EvaluationActionSeedCoordinates(
                    purpose=SeedPurpose.EVALUATION_ACTION,
                    evaluation_index=self.evaluation_index,
                    pair_index=self.pair_index,
                    game_in_pair=self.game_in_pair,
                    ply=ply,
                ),
            )
            for ply in range(len(self.action_seeds))
        )
        if (self.game_seed, self.search_seeds, self.action_seeds) != (
            expected_game,
            expected_search,
            expected_action,
        ):
            raise ValueError('Evaluation seed lineage does not match its deterministic coordinates.')
        return self


class EvaluationGameResult(FrozenModel):
    schema_version: Literal[1] = 1
    evaluation_id: UUID
    game_id: UUID
    pair_index: int = Field(ge=0)
    game_in_pair: int = Field(ge=0, le=1)
    requested_elapsed_seconds: int = Field(gt=0)
    published_checkpoint_elapsed_seconds: float = Field(ge=0)
    candidate: CandidateCheckpointIdentity
    opponent: EvaluationOpponentIdentity
    candidate_color: GoColor
    board_size: int = Field(ge=3)
    komi_half_points: int
    scoring_rule: Literal['area']
    ko_rule: Literal['positional_superko']
    suicide_rule: Literal['illegal']
    seed_lineage: EvaluationSeedLineage
    winner: GoColor | None
    candidate_score: float = Field(ge=0, le=1)
    termination: GameTermination
    plies: int = Field(ge=0)
    candidate_configured_simulations: int = Field(ge=0)
    candidate_actual_simulations: int = Field(ge=0)
    opponent_configured_simulations: int = Field(ge=0)
    opponent_actual_simulations: int = Field(ge=0)
    evaluation_wall_seconds: float = Field(ge=0)
    cost_category: Literal[EvaluationCostCategory.EVALUATION]

    @model_validator(mode='after')
    def validate_result(self) -> EvaluationGameResult:
        expected = (
            0.5
            if self.winner is None
            else float(
                (self.winner is GoColor.BLACK and self.candidate_color is GoColor.BLACK)
                or (self.winner is GoColor.WHITE and self.candidate_color is GoColor.WHITE)
            )
        )
        if self.candidate_score != expected:
            raise ValueError('Candidate score does not match winner and color.')
        if len(self.seed_lineage.search_seeds) != self.plies:
            raise ValueError('Evaluation seed streams must contain exactly one entry per ply.')
        if self.candidate_actual_simulations > self.candidate_configured_simulations:
            raise ValueError('Candidate actual simulations cannot exceed configured simulations.')
        if self.opponent_actual_simulations > self.opponent_configured_simulations:
            raise ValueError('Opponent actual simulations cannot exceed configured simulations.')
        if self.seed_lineage.pair_index != self.pair_index or self.seed_lineage.game_in_pair != self.game_in_pair:
            raise ValueError('Evaluation seed coordinates do not match the game identity.')
        match self.opponent:
            case RandomOpponentIdentity() if (
                self.opponent_configured_simulations != 0 or self.opponent_actual_simulations != 0
            ):
                raise ValueError('Random opponents cannot report search simulations.')
            case RandomOpponentIdentity() | CheckpointOpponentIdentity():
                pass
        return self


class EvaluationPairResult(FrozenModel):
    schema_version: Literal[1] = 1
    evaluation_id: UUID
    pair_index: int = Field(ge=0)
    games: tuple[EvaluationGameResult, EvaluationGameResult]

    @model_validator(mode='after')
    def validate_pair(self) -> EvaluationPairResult:
        if any(game.evaluation_id != self.evaluation_id for game in self.games):
            raise ValueError('Pair games must share the evaluation identity.')
        if tuple(game.game_in_pair for game in self.games) != (0, 1):
            raise ValueError('Pair games must be ordered zero then one.')
        if tuple(game.candidate_color for game in self.games) != (GoColor.BLACK, GoColor.WHITE):
            raise ValueError('Every pair must alternate candidate colors.')
        first, second = self.games
        if any(game.pair_index != self.pair_index for game in self.games):
            raise ValueError('Pair games must carry the enclosing pair index.')
        if (
            first.requested_elapsed_seconds != second.requested_elapsed_seconds
            or first.published_checkpoint_elapsed_seconds != second.published_checkpoint_elapsed_seconds
        ):
            raise ValueError('Pair games must share requested and actual checkpoint timing evidence.')
        if (
            first.seed_lineage.root_seed != second.seed_lineage.root_seed
            or first.seed_lineage.evaluation_index != second.seed_lineage.evaluation_index
        ):
            raise ValueError('Pair games must share root seed and evaluation index.')
        common = (
            first.candidate,
            first.opponent,
            first.board_size,
            first.komi_half_points,
            first.scoring_rule,
            first.ko_rule,
            first.suicide_rule,
        )
        other = (
            second.candidate,
            second.opponent,
            second.board_size,
            second.komi_half_points,
            second.scoring_rule,
            second.ko_rule,
            second.suicide_rule,
        )
        if common != other:
            raise ValueError('Paired games must share checkpoints and exact Go rules.')
        return self
