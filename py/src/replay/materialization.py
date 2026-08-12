from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import TypeVar

from src.games.contracts import GameStateContract, WdlTarget
from src.replay.contracts import (
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    IneligibleNextPolicyTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.self_play.completed_game import CompletedSelfPlayGame, SearchObservation, TerminationReason
from src.self_play.policy import ordered_search_visits
from src.training.targets import NextPolicyHeadLayout, RemainingGameLengthHeadLayout, TrainingTargetLayout


PositionT = TypeVar('PositionT')


@dataclass(frozen=True)
class PolicyRetention:
    policy: SparsePolicyTarget
    truncated: bool
    retained_visit_mass: int
    discarded_visit_mass: int


@dataclass(frozen=True)
class MaterializedGame:
    samples: tuple[ReplaySample, ...]
    policies_truncated: int
    retained_visit_mass: int
    discarded_visit_mass: int


def retain_policy(observation: SearchObservation, maximum_entries: int) -> PolicyRetention:
    ordered = ordered_search_visits(observation)
    if not ordered:
        raise ValueError(f'Observation at ply {observation.ply} has no policy-target visits.')
    retained = ordered[:maximum_entries]
    discarded = ordered[maximum_entries:]
    return PolicyRetention(
        policy=SparsePolicyTarget(visits=retained),
        truncated=bool(discarded),
        retained_visit_mass=sum(visit.visit_count for visit in retained),
        discarded_visit_mass=sum(visit.visit_count for visit in discarded),
    )


def materialize_completed_game(
    game: CompletedSelfPlayGame,
    state: GameStateContract[PositionT],
    targets: TrainingTargetLayout,
    maximum_policy_entries: int,
) -> MaterializedGame:
    if targets.action_size != state.action_size:
        raise ValueError('Training target action count does not match the game state contract.')
    if maximum_policy_entries <= 0:
        raise ValueError('Maximum retained policy entries must be positive.')

    positions = _reconstruct_trajectory(game, state)
    observations = {observation.ply: observation for observation in game.observations}
    _validate_result(game, state, positions[-1])

    samples: list[ReplaySample] = []
    policies_truncated = 0
    retained_visit_mass = 0
    discarded_visit_mass = 0
    final_player = state.current_player(positions[-1])
    for observation in game.observations:
        if not observation.full_search:
            continue
        primary = retain_policy(observation, maximum_policy_entries)
        policies_truncated += int(primary.truncated)
        retained_visit_mass += primary.retained_visit_mass
        discarded_visit_mass += primary.discarded_visit_mass

        auxiliary_targets = []
        for head in targets.auxiliary_heads:
            match head:
                case NextPolicyHeadLayout(ply_offset=ply_offset):
                    future_observation = observations.get(observation.ply + ply_offset)
                    if future_observation is None:
                        auxiliary_targets.append(IneligibleNextPolicyTarget())
                        continue
                    auxiliary = retain_policy(future_observation, maximum_policy_entries)
                    policies_truncated += int(auxiliary.truncated)
                    retained_visit_mass += auxiliary.retained_visit_mass
                    discarded_visit_mass += auxiliary.discarded_visit_mass
                    auxiliary_targets.append(EligibleNextPolicyTarget(policy=auxiliary.policy))
                case RemainingGameLengthHeadLayout(normalization_scale=normalization_scale):
                    remaining_plies = len(game.action_ids) - observation.ply
                    auxiliary_targets.append(
                        EligibleRemainingGameLengthTarget(normalized_length=remaining_plies / normalization_scale)
                    )

        position = positions[observation.ply]
        position_wdl = game.final_wdl if state.current_player(position) == final_player else game.final_wdl.reversed()
        samples.append(
            ReplaySample(
                encoded_state=state.encode_network_input(position),
                policy=primary.policy,
                wdl_target=position_wdl,
                root_value=observation.root_value,
                auxiliary_targets=tuple(auxiliary_targets),
                sample_weight=observation.sample_weight,
                source_model_generation=observation.model_generation,
                source_created_at_seconds=game.created_at_seconds,
            )
        )
    return MaterializedGame(
        samples=tuple(samples),
        policies_truncated=policies_truncated,
        retained_visit_mass=retained_visit_mass,
        discarded_visit_mass=discarded_visit_mass,
    )


def _reconstruct_trajectory(
    game: CompletedSelfPlayGame,
    state: GameStateContract[PositionT],
) -> tuple[PositionT, ...]:
    positions = [state.initial_position()]
    observations = {observation.ply: observation for observation in game.observations}
    for ply, action_id in enumerate(game.action_ids):
        position = positions[-1]
        legal_actions = state.legal_action_ids(position)
        if action_id not in legal_actions:
            raise ValueError(f'Played action {action_id} is illegal at ply {ply}.')
        observation = observations.get(ply)
        if observation is not None:
            if observation.selected_action_id != action_id:
                raise ValueError(f'Observed action does not match the played action at ply {ply}.')
            if any(visit.action_id not in legal_actions for visit in observation.policy_target_visits):
                raise ValueError(f'Observation at ply {ply} contains an illegal action.')
        positions.append(state.child_position(position, action_id))
    for observation in game.observations:
        legal_actions = state.legal_action_ids(positions[observation.ply])
        if any(visit.action_id not in legal_actions for visit in observation.policy_target_visits):
            raise ValueError(f'Observation at ply {observation.ply} contains an illegal action.')
        if observation.highest_visited_child_action_id not in legal_actions:
            raise ValueError(f'Observation at ply {observation.ply} has an illegal highest-visited child.')
    return tuple(positions)


def _validate_result(
    game: CompletedSelfPlayGame,
    state: GameStateContract[PositionT],
    final_position: PositionT,
) -> None:
    match game.termination_reason:
        case TerminationReason.NATURAL:
            expected = state.natural_terminal_wdl(final_position)
            if expected is None:
                raise ValueError('Naturally terminated game does not end in a terminal position.')
        case TerminationReason.MAXIMUM_PLIES | TerminationReason.ADJUDICATION:
            expected = state.adjudicated_wdl(final_position, game.termination_reason)
        case TerminationReason.RESIGNATION:
            if state.natural_terminal_wdl(final_position) is not None:
                raise ValueError('Resigned game should have been recorded as a natural terminal result.')
            if not _same_wdl(game.final_wdl, WdlTarget(win=0.0, draw=0.0, loss=1.0)):
                raise ValueError('Resignation must record a loss for the player to move.')
            return
    if not _same_wdl(expected, game.final_wdl):
        raise ValueError('Completed-game result disagrees with the reconstructed final position.')


def _same_wdl(first: WdlTarget, second: WdlTarget) -> bool:
    return all(
        isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
        for left, right in zip(
            (first.win, first.draw, first.loss),
            (second.win, second.draw, second.loss),
        )
    )
