from __future__ import annotations

import pytest
from src.experiment.configuration import load_experiment_configuration
from src.training.tensorboard import scheduled_settings_at
from test_helpers.configuration_paths import REPOSITORY_CONFIG_DIRECTORY


def test_scheduled_settings_include_every_generation_schedule() -> None:
    configuration = load_experiment_configuration(
        REPOSITORY_CONFIG_DIRECTORY / 'baselines' / 'vast-go-9x9-2gpu-4h.yaml'
    )

    settings = {setting.tag: setting.value for setting in scheduled_settings_at(configuration, 25)}

    assert set(settings) == {
        'settings/training/learning_rate',
        'settings/replay/capacity',
        'settings/self_play/baseline_visits',
        'settings/self_play/dirichlet_epsilon',
        'settings/self_play/dirichlet_alpha',
        'settings/self_play/exploration_constant',
        'settings/self_play/retained_root_visit_fraction',
        'settings/self_play/greedy_after_ply',
        'settings/self_play/starting_temperature',
        'settings/self_play/final_temperature',
        'settings/self_play/primary_sample_weight',
        'settings/self_play/first_play_urgency_reduction',
        'settings/training/policy_loss_weight',
        'settings/training/value_loss_weight',
        'settings/training/root_value_blend',
        'settings/training/value_discount_per_ply',
        'settings/training/auxiliary/0-next-policy-ply-1/loss_weight',
        'settings/training/auxiliary/1-remaining-game-length/loss_weight',
    }
    assert settings['settings/training/learning_rate'] == pytest.approx(0.0085)
    assert settings['settings/replay/capacity'] == 500_000
    assert settings['settings/self_play/baseline_visits'] == 160


def test_scheduled_settings_resolve_an_automatic_exploration_constant() -> None:
    # 'auto' is a literal rather than a schedule, so reading it as one survives configuration
    # validation and only fails once a run publishes its settings.
    configuration = load_experiment_configuration(
        REPOSITORY_CONFIG_DIRECTORY / 'production' / 'vast-chess-8gpu-v101-fp16-big-wide-value.yaml'
    )
    assert configuration.chess.self_play.search.exploration_constant == 'auto'

    settings = {setting.tag: setting.value for setting in scheduled_settings_at(configuration, 25)}

    # Generation 25 sits on the 400-visit stage, which the AlphaZero schedule puts at 1.2702.
    assert settings['settings/self_play/exploration_constant'] == pytest.approx(1.2702, abs=1e-4)
