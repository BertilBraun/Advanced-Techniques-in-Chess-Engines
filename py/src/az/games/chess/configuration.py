from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from src.az.config.artifacts import CheckpointArtifactReference
from src.az.config.base import FrozenModel, Sha256
from src.az.config.search import (
    ConstantTemperature,
    DisabledRootExploration,
    DisabledTreeReuse,
    FixedSearchBudget,
    FullBudgetStopping,
    SearchConfiguration,
)
from src.az.config.training import (
    LearningRateConfiguration,
    OptimizerConfiguration,
    ReplayCreditConfiguration,
)


CHESS_LAYER_A_SCHEMA_VERSION = 1
CHESS_LAYER_B_SCHEMA_VERSION = 1


class ChessGameConfiguration(FrozenModel):
    kind: Literal["chess"]
    variant: Literal["standard"]
    input_encoding: Literal["canonical_8x8_history_v1"]
    action_encoding: Literal["canonical_1880_move_map_v1"]
    history_length: PositiveInt
    repetition_draw_count: Literal[3]
    halfmove_draw_ply_count: Literal[100]
    insufficient_material_draw: Literal[True]
    safety_ply_cap: PositiveInt
    perspective: Literal["side_to_move"]
    symmetry_group: Literal["identity"]

    @property
    def action_count(self) -> int:
        return 1_880


class ResidualChessModelConfiguration(FrozenModel):
    family: Literal["residual_chess"]
    channels: PositiveInt
    residual_blocks: PositiveInt
    policy_channels: PositiveInt
    value_hidden_size: PositiveInt
    normalization: Literal["batch"]
    activation: Literal["relu"]
    value_head: Literal["wdl"]


class FixedChessModelSchedule(FrozenModel):
    kind: Literal["fixed"]
    architecture: ResidualChessModelConfiguration


class ChessModelStage(FrozenModel):
    start_elapsed_seconds: int = Field(ge=0)
    architecture: ResidualChessModelConfiguration


class ProgressiveChessModelSchedule(FrozenModel):
    kind: Literal["progressive"]
    stages: tuple[ChessModelStage, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_stages(self) -> ProgressiveChessModelSchedule:
        starts = tuple(stage.start_elapsed_seconds for stage in self.stages)
        if starts[0] != 0 or tuple(sorted(set(starts))) != starts:
            raise ValueError(
                "Progressive chess model stages must start at zero and increase strictly."
            )
        return self


ChessModelSchedule = Annotated[
    FixedChessModelSchedule | ProgressiveChessModelSchedule,
    Field(discriminator="kind"),
]


class ChessModelConfiguration(FrozenModel):
    schedule: ChessModelSchedule


class ChessObjectiveConfiguration(FrozenModel):
    kind: Literal["chess_policy_wdl"]
    policy_loss_weight: PositiveFloat
    value_loss_weight: PositiveFloat
    l2_regularization_weight: float = Field(ge=0)


class ChessReplayConfiguration(FrozenModel):
    kind: Literal["chess_layer_a_b"]
    capacity_positions: PositiveInt
    layer_a_directory: PurePosixPath
    layer_b_directory: PurePosixPath
    maximum_layer_a_games_per_shard: PositiveInt
    maximum_layer_b_positions_per_shard: PositiveInt
    layer_a_schema_version: Literal[CHESS_LAYER_A_SCHEMA_VERSION]
    layer_b_schema_version: Literal[CHESS_LAYER_B_SCHEMA_VERSION]
    compression: Literal["zstd", "none"]
    sampling: Literal["uniform"]
    credits: ReplayCreditConfiguration


class ChessTrainingConfiguration(FrozenModel):
    global_batch_size: PositiveInt
    local_batch_size: PositiveInt
    maximum_optimizer_steps: PositiveInt
    optimizer: OptimizerConfiguration
    learning_rate_schedule: LearningRateConfiguration
    precision: Literal["float32", "bfloat16", "float16"]
    objective: ChessObjectiveConfiguration
    checkpoint_every_optimizer_steps: PositiveInt
    gradient_clip_norm: PositiveFloat


class RandomChessOpponent(FrozenModel):
    kind: Literal["random"]


class CheckpointChessOpponent(FrozenModel):
    kind: Literal["checkpoint"]
    checkpoint: CheckpointArtifactReference


class StockfishEngineConfiguration(FrozenModel):
    executable_path: PurePosixPath
    executable_sha256: Sha256
    protocol: Literal["uci"]
    threads: PositiveInt
    hash_mebibytes: PositiveInt
    ponder: Literal[False]


class StockfishSkillOpponent(FrozenModel):
    kind: Literal["stockfish_skill"]
    engine: StockfishEngineConfiguration
    skill_level: int = Field(ge=0, le=20)


class StockfishNodesOpponent(FrozenModel):
    kind: Literal["stockfish_nodes"]
    engine: StockfishEngineConfiguration
    nodes_per_move: PositiveInt


ChessOpponentConfiguration = Annotated[
    RandomChessOpponent
    | CheckpointChessOpponent
    | StockfishSkillOpponent
    | StockfishNodesOpponent,
    Field(discriminator="kind"),
]


class ChessEvaluationSuite(FrozenModel):
    kind: Literal["chess_ladder"]
    opponents: tuple[ChessOpponentConfiguration, ...] = Field(min_length=1)
    alternate_colors: Literal[True]
    opening_source: Literal["initial_position", "paired_openings"]


class ChessEvaluationConfiguration(FrozenModel):
    search: SearchConfiguration
    checkpoint_elapsed_seconds: tuple[PositiveInt, ...]
    paired_games_per_checkpoint: PositiveInt
    bootstrap_samples: PositiveInt
    confidence_method: Literal["paired_bootstrap"]
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_seed: int = Field(ge=0, le=2**63 - 1)
    suite: ChessEvaluationSuite

    @model_validator(mode="after")
    def validate_evaluation(self) -> ChessEvaluationConfiguration:
        if self.paired_games_per_checkpoint % 2 != 0:
            raise ValueError("Paired evaluation game count must be even.")
        if (
            tuple(sorted(set(self.checkpoint_elapsed_seconds)))
            != self.checkpoint_elapsed_seconds
        ):
            raise ValueError(
                "Evaluation checkpoint times must be unique and strictly increasing."
            )
        match (
            self.search.budget,
            self.search.stopping,
            self.search.root_exploration,
            self.search.temperature,
            self.search.tree_reuse,
        ):
            case (
                FixedSearchBudget(),
                FullBudgetStopping(),
                DisabledRootExploration(),
                ConstantTemperature(temperature=0.0),
                DisabledTreeReuse(),
            ):
                return self
            case _:
                raise ValueError(
                    "Common evaluation search requires a fixed full budget, disabled root noise, "
                    "zero action temperature, and disabled tree reuse."
                )


def validate_chess_experiment_compatibility(
    objective: ChessObjectiveConfiguration,
    optimizer_weight_decay: float,
) -> None:
    if objective.l2_regularization_weight > 0 and optimizer_weight_decay > 0:
        raise ValueError(
            "Objective L2 and optimizer weight decay are intentionally mutually exclusive experimental choices."
        )
