from pathlib import Path

import pytest

from src.experiment.configuration import load_experiment_configuration
from src.training.tensorboard import scheduled_settings_at


def test_scheduled_settings_include_every_generation_schedule() -> None:
    configuration = load_experiment_configuration(Path('configs/baselines/vast-go-9x9-2gpu-4h.yaml'))

    settings = {setting.tag: setting.value for setting in scheduled_settings_at(configuration, 25)}

    assert set(settings) == {
        'settings/training/learning_rate',
        'settings/replay/capacity',
        'settings/self_play/full_searches',
        'settings/self_play/fast_searches',
        'settings/self_play/dirichlet_epsilon',
        'settings/self_play/dirichlet_alpha',
        'settings/self_play/exploration_constant',
        'settings/self_play/full_search_probability',
        'settings/self_play/retained_root_visit_fraction',
        'settings/self_play/greedy_after_ply',
        'settings/self_play/starting_temperature',
        'settings/self_play/final_temperature',
        'settings/self_play/primary_sample_weight',
        'settings/self_play/first_play_urgency_reduction',
        'settings/training/policy_loss_weight',
        'settings/training/value_loss_weight',
        'settings/training/root_value_blend',
        'settings/training/auxiliary/0-next-policy-ply-1/loss_weight',
        'settings/training/auxiliary/1-remaining-game-length/loss_weight',
    }
    assert settings['settings/training/learning_rate'] == pytest.approx(0.0085)
    assert settings['settings/replay/capacity'] == 500_000
    assert settings['settings/self_play/full_searches'] == 160
    assert settings['settings/self_play/full_search_probability'] == pytest.approx(0.25)
