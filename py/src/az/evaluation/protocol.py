from __future__ import annotations

import io
import random
import time
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID, uuid5

import torch

import az_go_native as native
from src.az.config.search import (
    FixedSearchBudget,
    ParentValueFpu,
    ReducedParentValueFpu,
    SearchConfiguration,
    VisitedChildMeanFpu,
)
from src.az.config.seeds import (
    EvaluationActionSeedCoordinates,
    EvaluationGameSeedCoordinates,
    EvaluationSearchSeedCoordinates,
    SEED_DERIVATION_VERSION,
    SeedPurpose,
    derive_seed,
)
from src.az.evaluation.models import (
    CandidateCheckpointIdentity,
    CheckpointOpponentIdentity,
    EvaluationCostCategory,
    EvaluationGameResult,
    EvaluationOpponentIdentity,
    EvaluationPairResult,
    EvaluationSeedLineage,
    GoColor,
)
from src.az.evaluation.checkpoints import EvaluationModelArtifactRepository
from src.az.evaluation.storage import EvaluationResultRepository
from src.az.games.go.configuration import GoGameConfiguration, ResidualGoModelConfiguration
from src.az.games.go.model import ResidualGoModel
from src.az.replay.envelope import GameTermination
from src.az.inference.go_batching import GoInferenceBatchBroker
from src.az.config.serialization import canonical_json
from src.az.config.serialization import model_sha256
from src.az.config.base import FrozenModel, Sha256
from pydantic import Field, model_validator


@dataclass(frozen=True)
class MoveSelection:
    action: int
    configured_simulations: int
    actual_simulations: int


class EvaluationPlayer(Protocol):
    @property
    def search_configuration_sha256(self) -> str | None: ...

    @property
    def checkpoint_identity(self) -> CandidateCheckpointIdentity | None: ...

    def select_action(self, state: native.GoState, search_seed: int, action_seed: int) -> MoveSelection: ...


class CheckpointModelLoader(Protocol):
    def load(self, identity: CandidateCheckpointIdentity) -> LoadedEvaluationModel: ...


@dataclass(frozen=True)
class LoadedEvaluationModel:
    identity: CandidateCheckpointIdentity
    model: ResidualGoModel


class RepositoryCheckpointModelLoader:
    def __init__(
        self,
        repository: EvaluationModelArtifactRepository,
        game: GoGameConfiguration,
        model: ResidualGoModelConfiguration,
        device: torch.device,
    ) -> None:
        self._repository = repository
        self._game = game
        self._model = model
        self._device = device

    def load(self, identity: CandidateCheckpointIdentity) -> LoadedEvaluationModel:
        model_artifact = self._repository.load(identity)
        model = ResidualGoModel(self._game, self._model)
        model.load_state_dict(
            torch.load(io.BytesIO(model_artifact), map_location=self._device, weights_only=True),
            strict=True,
        )
        return LoadedEvaluationModel(identity=identity, model=model)


class RandomGoEvaluationPlayer:
    @property
    def search_configuration_sha256(self) -> None:
        return None

    @property
    def checkpoint_identity(self) -> None:
        return None

    def select_action(self, state: native.GoState, search_seed: int, action_seed: int) -> MoveSelection:
        del search_seed
        legal = state.legal_actions()
        action = legal[random.Random(action_seed).randrange(len(legal))]
        return MoveSelection(action=action, configured_simulations=0, actual_simulations=0)


class NativeCheckpointEvaluationPlayer:
    def __init__(
        self,
        broker: GoInferenceBatchBroker,
        search: SearchConfiguration,
        loaded_model: LoadedEvaluationModel,
    ) -> None:
        match search.budget:
            case FixedSearchBudget(simulations=simulations):
                self._simulation_cap = simulations
            case _:
                raise ValueError('Evaluation player requires a fixed search budget.')
        self._broker = broker
        self._search = search
        self._checkpoint_identity = loaded_model.identity
        if broker.model is not loaded_model.model:
            raise ValueError('Evaluation broker does not own the declared loaded checkpoint model.')

    @property
    def search_configuration_sha256(self) -> str:
        return model_sha256(self._search)

    @property
    def checkpoint_identity(self) -> CandidateCheckpointIdentity:
        return self._checkpoint_identity

    def select_action(self, state: native.GoState, search_seed: int, action_seed: int) -> MoveSelection:
        match self._search.fpu:
            case ParentValueFpu():
                fpu_policy, reduction, fallback = native.FpuPolicy.PARENT_VALUE, 0.0, 0.0
            case ReducedParentValueFpu(reduction=reduction):
                fpu_policy, fallback = native.FpuPolicy.REDUCED_PARENT_VALUE, 0.0
            case VisitedChildMeanFpu(no_visited_child_value=fallback):
                fpu_policy, reduction = native.FpuPolicy.VISITED_CHILD_MEAN, 0.0
        configuration = native.FixedPuctConfiguration(
            simulation_cap=self._simulation_cap,
            exploration_constant=self._search.algorithm.exploration_constant,
            backup_discount=self._search.backup_discount,
            no_visited_child_value=fallback,
            action_temperature=0.0,
            root_noise_seed=search_seed,
            action_sampling_seed=action_seed,
            root_noise=native.RootNoiseConfiguration(False, 1.0, 0.0),
            tree_reuse=False,
            fpu_policy=fpu_policy,
            fpu_reduction=reduction,
            adaptive_stopping=native.AdaptiveStoppingConfiguration(False, 1, 1, 1.0, 1.0),
            budget_class=native.SearchBudgetClass.FIXED,
            policy_target_weight=0.0,
        )
        result = native.search_go_fixed(state, self._broker.evaluate, configuration)
        if result.selected_action is None:
            raise AssertionError('A nonterminal evaluation state must produce an action.')
        return MoveSelection(
            action=result.selected_action,
            configured_simulations=self._simulation_cap,
            actual_simulations=result.telemetry.actual_simulations,
        )


class PairedEvaluationSpecification(FrozenModel):
    evaluation_id: UUID
    run_id: UUID
    resolved_configuration_sha256: Sha256
    common_search_sha256: Sha256
    evaluation_index: int = Field(ge=0)
    root_seed: int = Field(ge=0, le=2**63 - 1)
    requested_elapsed_seconds: int = Field(gt=0)
    published_checkpoint_elapsed_seconds: float = Field(ge=0)
    candidate: CandidateCheckpointIdentity
    opponent: EvaluationOpponentIdentity
    game: GoGameConfiguration

    @model_validator(mode='after')
    def validate_identity(self) -> PairedEvaluationSpecification:
        if self.evaluation_id != derive_evaluation_id(
            self.run_id,
            self.resolved_configuration_sha256,
            self.common_search_sha256,
            self.evaluation_index,
            self.requested_elapsed_seconds,
            self.candidate,
            self.opponent,
            self.game,
        ):
            raise ValueError('Evaluation identity does not match its stable specification.')
        return self


def derive_evaluation_id(
    run_id: UUID,
    resolved_configuration_sha256: str,
    common_search_sha256: str,
    evaluation_index: int,
    requested_elapsed_seconds: int,
    candidate: CandidateCheckpointIdentity,
    opponent: EvaluationOpponentIdentity,
    game: GoGameConfiguration,
) -> UUID:
    material = (
        f'{resolved_configuration_sha256}\0'
        f'{common_search_sha256}\0'
        f'{evaluation_index}\0'
        f'{requested_elapsed_seconds}\0'
        f'{canonical_json(candidate)}\0'
        f'{canonical_json(opponent)}\0'
        f'{canonical_json(game)}'
    )
    return uuid5(run_id, material)


class PairedGoEvaluator:
    def __init__(
        self,
        specification: PairedEvaluationSpecification,
        candidate_player: EvaluationPlayer,
        opponent_player: EvaluationPlayer,
        repository: EvaluationResultRepository,
    ) -> None:
        self._specification = specification
        self._candidate_player = candidate_player
        self._opponent_player = opponent_player
        self._repository = repository
        if candidate_player.search_configuration_sha256 != specification.common_search_sha256:
            raise ValueError('Candidate player does not use the declared common evaluation search.')
        if candidate_player.checkpoint_identity != specification.candidate:
            raise ValueError('Candidate player model does not match the declared checkpoint.')
        match specification.opponent:
            case CheckpointOpponentIdentity() if (
                opponent_player.search_configuration_sha256 != specification.common_search_sha256
            ):
                raise ValueError('Checkpoint opponent does not use the declared common evaluation search.')
            case CheckpointOpponentIdentity(checkpoint=checkpoint) if opponent_player.checkpoint_identity != checkpoint:
                raise ValueError('Checkpoint opponent model does not match the declared checkpoint.')
            case CheckpointOpponentIdentity():
                pass
            case _ if opponent_player.search_configuration_sha256 is not None:
                raise ValueError('Non-checkpoint opponent unexpectedly declares checkpoint search.')
            case _:
                pass

    def evaluate_pair(self, pair_index: int) -> EvaluationPairResult:
        games = tuple(self._evaluate_or_load(pair_index, game_in_pair) for game_in_pair in (0, 1))
        return EvaluationPairResult(
            evaluation_id=self._specification.evaluation_id,
            pair_index=pair_index,
            games=(games[0], games[1]),
        )

    def _evaluate_or_load(self, pair_index: int, game_in_pair: int) -> EvaluationGameResult:
        existing = self._repository.load(self._specification.evaluation_id, pair_index, game_in_pair)
        if existing is not None:
            self._validate_existing(existing, pair_index, game_in_pair)
            return existing
        return self._repository.publish(self._play(pair_index, game_in_pair))

    def _validate_existing(
        self,
        result: EvaluationGameResult,
        pair_index: int,
        game_in_pair: int,
    ) -> None:
        specification = self._specification
        expected = (
            specification.evaluation_id,
            pair_index,
            game_in_pair,
            specification.requested_elapsed_seconds,
            specification.published_checkpoint_elapsed_seconds,
            specification.common_search_sha256,
            specification.candidate,
            specification.opponent,
            specification.game.board_size,
            specification.game.komi_half_points,
            specification.game.scoring_rule,
            specification.game.ko_rule,
            specification.game.suicide_rule,
            specification.root_seed,
            specification.evaluation_index,
        )
        actual = (
            result.evaluation_id,
            result.pair_index,
            result.game_in_pair,
            result.requested_elapsed_seconds,
            result.published_checkpoint_elapsed_seconds,
            result.common_search_sha256,
            result.candidate,
            result.opponent,
            result.board_size,
            result.komi_half_points,
            result.scoring_rule,
            result.ko_rule,
            result.suicide_rule,
            result.seed_lineage.root_seed,
            result.seed_lineage.evaluation_index,
        )
        if actual != expected:
            raise ValueError('Stored evaluation result is incompatible with the active specification.')

    def _play(self, pair_index: int, game_in_pair: int) -> EvaluationGameResult:
        specification = self._specification
        game_seed = derive_seed(
            specification.root_seed,
            EvaluationGameSeedCoordinates(
                purpose=SeedPurpose.EVALUATION_GAME,
                evaluation_index=specification.evaluation_index,
                pair_index=pair_index,
                game_in_pair=game_in_pair,
            ),
        )
        candidate_is_black = game_in_pair == 0
        rules = native.GoRules(
            specification.game.board_size,
            specification.game.komi_half_points,
            specification.game.safety_ply_cap,
            specification.game.history_length,
        )
        state = native.GoState(rules)
        candidate_configured = candidate_actual = opponent_configured = opponent_actual = 0
        search_seeds: list[int] = []
        action_seeds: list[int] = []
        started = time.perf_counter()
        while not state.is_terminal:
            search_seed = derive_seed(
                specification.root_seed,
                EvaluationSearchSeedCoordinates(
                    purpose=SeedPurpose.EVALUATION_SEARCH,
                    evaluation_index=specification.evaluation_index,
                    pair_index=pair_index,
                    game_in_pair=game_in_pair,
                    ply=state.ply,
                ),
            )
            action_seed = derive_seed(
                specification.root_seed,
                EvaluationActionSeedCoordinates(
                    purpose=SeedPurpose.EVALUATION_ACTION,
                    evaluation_index=specification.evaluation_index,
                    pair_index=pair_index,
                    game_in_pair=game_in_pair,
                    ply=state.ply,
                ),
            )
            search_seeds.append(search_seed)
            action_seeds.append(action_seed)
            black_to_move = state.current_player == native.Player.BLACK
            candidate_to_move = black_to_move == candidate_is_black
            player = self._candidate_player if candidate_to_move else self._opponent_player
            move = player.select_action(state, search_seed, action_seed)
            if candidate_to_move:
                candidate_configured += move.configured_simulations
                candidate_actual += move.actual_simulations
            else:
                opponent_configured += move.configured_simulations
                opponent_actual += move.actual_simulations
            state.apply(move.action)
        wall_seconds = time.perf_counter() - started
        terminal = state.terminal_result()
        winner = (
            None
            if terminal.winner is None
            else GoColor.BLACK
            if terminal.winner == native.Player.BLACK
            else GoColor.WHITE
        )
        candidate_color = GoColor.BLACK if candidate_is_black else GoColor.WHITE
        candidate_score = 0.5 if winner is None else float(winner is candidate_color)
        termination = (
            GameTermination.SAFETY_PLY_CAP
            if terminal.reason == native.TerminationReason.SAFETY_PLY_CAP
            else GameTermination.TWO_CONSECUTIVE_PASSES
        )
        return EvaluationGameResult(
            evaluation_id=specification.evaluation_id,
            game_id=uuid5(specification.evaluation_id, f'pair:{pair_index}:game:{game_in_pair}'),
            pair_index=pair_index,
            game_in_pair=game_in_pair,
            requested_elapsed_seconds=specification.requested_elapsed_seconds,
            published_checkpoint_elapsed_seconds=specification.published_checkpoint_elapsed_seconds,
            common_search_sha256=specification.common_search_sha256,
            candidate=specification.candidate,
            opponent=specification.opponent,
            candidate_color=candidate_color,
            board_size=specification.game.board_size,
            komi_half_points=specification.game.komi_half_points,
            scoring_rule=specification.game.scoring_rule,
            ko_rule=specification.game.ko_rule,
            suicide_rule=specification.game.suicide_rule,
            seed_lineage=EvaluationSeedLineage(
                derivation_version=SEED_DERIVATION_VERSION,
                root_seed=specification.root_seed,
                evaluation_index=specification.evaluation_index,
                pair_index=pair_index,
                game_in_pair=game_in_pair,
                game_seed=game_seed,
                search_seeds=tuple(search_seeds),
                action_seeds=tuple(action_seeds),
            ),
            winner=winner,
            candidate_score=candidate_score,
            termination=termination,
            plies=state.ply,
            candidate_configured_simulations=candidate_configured,
            candidate_actual_simulations=candidate_actual,
            opponent_configured_simulations=opponent_configured,
            opponent_actual_simulations=opponent_actual,
            evaluation_wall_seconds=wall_seconds,
            cost_category=EvaluationCostCategory.EVALUATION,
        )
