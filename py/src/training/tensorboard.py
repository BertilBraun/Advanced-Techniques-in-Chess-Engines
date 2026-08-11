from __future__ import annotations

from dataclasses import dataclass

from src.experiment.configuration import ExperimentConfiguration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.go.configuration import GoExperimentConfiguration
from src.self_play.configuration import (
    RandomOpeningStartConfiguration,
    ReducedParentValueFirstPlayUrgencyConfiguration,
    SelfPlayConfiguration,
)
from src.training.configuration import TrainingObjectiveConfiguration
from src.training.targets import NextPolicyTargetConfiguration, RemainingGameLengthTargetConfiguration


@dataclass(frozen=True)
class ScheduledSetting:
    tag: str
    value: float | int


def scheduled_settings_at(
    configuration: ExperimentConfiguration,
    model_generation: int,
) -> tuple[ScheduledSetting, ...]:
    training = configuration.training
    settings = [
        ScheduledSetting('settings/training/learning_rate', training.trainer.learning_rate.value_at(model_generation)),
        ScheduledSetting(
            'settings/replay/capacity',
            training.lifecycle.replay.capacity.value_at(model_generation),
        ),
    ]
    match configuration:
        case ChessExperimentConfiguration(chess=chess):
            self_play = chess.self_play
            objective = chess.objective
            if self_play.maximum_game_plies is not None:
                settings.append(
                    ScheduledSetting(
                        'settings/self_play/maximum_game_plies',
                        self_play.maximum_game_plies.value_at(model_generation),
                    )
                )
        case GoExperimentConfiguration(go=go):
            self_play = go.self_play
            objective = go.objective
    settings.extend(_self_play_settings_at(self_play, model_generation))
    settings.extend(_objective_settings_at(objective, model_generation))
    return tuple(settings)


def _self_play_settings_at(
    configuration: SelfPlayConfiguration,
    model_generation: int,
) -> tuple[ScheduledSetting, ...]:
    search = configuration.search
    settings = [
        ScheduledSetting('settings/self_play/full_searches', search.full_searches.value_at(model_generation)),
        ScheduledSetting('settings/self_play/fast_searches', search.fast_searches.value_at(model_generation)),
        ScheduledSetting('settings/self_play/dirichlet_epsilon', search.dirichlet_epsilon.value_at(model_generation)),
        ScheduledSetting('settings/self_play/dirichlet_alpha', search.dirichlet_alpha.value_at(model_generation)),
        ScheduledSetting(
            'settings/self_play/exploration_constant', search.exploration_constant.value_at(model_generation)
        ),
        ScheduledSetting(
            'settings/self_play/full_search_probability',
            configuration.full_search_probability.value_at(model_generation),
        ),
        ScheduledSetting(
            'settings/self_play/retained_root_visit_fraction',
            configuration.retained_root_visit_fraction.value_at(model_generation),
        ),
        ScheduledSetting(
            'settings/self_play/greedy_after_ply', configuration.greedy_after_ply.value_at(model_generation)
        ),
        ScheduledSetting(
            'settings/self_play/starting_temperature', configuration.starting_temperature.value_at(model_generation)
        ),
        ScheduledSetting(
            'settings/self_play/final_temperature', configuration.final_temperature.value_at(model_generation)
        ),
        ScheduledSetting(
            'settings/self_play/primary_sample_weight',
            configuration.primary_sample_weight.value_at(model_generation),
        ),
    ]
    if configuration.force_fast_search_after_ply is not None:
        settings.append(
            ScheduledSetting(
                'settings/self_play/force_fast_search_after_ply',
                configuration.force_fast_search_after_ply.value_at(model_generation),
            )
        )
    match search.first_play_urgency:
        case ReducedParentValueFirstPlayUrgencyConfiguration(reduction=reduction):
            settings.append(
                ScheduledSetting(
                    'settings/self_play/first_play_urgency_reduction',
                    reduction.value_at(model_generation),
                )
            )
    match configuration.start_position:
        case RandomOpeningStartConfiguration(maximum_plies=maximum_plies):
            settings.append(
                ScheduledSetting(
                    'settings/self_play/maximum_opening_plies',
                    maximum_plies.value_at(model_generation),
                )
            )
    return tuple(settings)


def _objective_settings_at(
    configuration: TrainingObjectiveConfiguration,
    model_generation: int,
) -> tuple[ScheduledSetting, ...]:
    settings = [
        ScheduledSetting(
            'settings/training/policy_loss_weight',
            configuration.policy_loss_weight.value_at(model_generation),
        ),
        ScheduledSetting(
            'settings/training/value_loss_weight',
            configuration.value_loss_weight.value_at(model_generation),
        ),
        ScheduledSetting(
            'settings/training/root_value_blend',
            configuration.root_value_blend.value_at(model_generation),
        ),
    ]
    for index, target in enumerate(configuration.auxiliary_targets):
        match target:
            case NextPolicyTargetConfiguration(ply_offset=ply_offset, loss_weight=loss_weight):
                name = f'{index}-next-policy-ply-{ply_offset}'
            case RemainingGameLengthTargetConfiguration(loss_weight=loss_weight):
                name = f'{index}-remaining-game-length'
        settings.append(
            ScheduledSetting(
                f'settings/training/auxiliary/{name}/loss_weight',
                loss_weight.value_at(model_generation),
            )
        )
    return tuple(settings)
