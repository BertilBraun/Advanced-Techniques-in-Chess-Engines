from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.configuration import load_experiment_configuration
from src.experiment_queue.configuration import load_queue_configuration

CONFIGURATION_ROOT = Path(__file__).resolve().parents[1] / 'configs'
EXPERIMENT_FILES = sorted(path for path in CONFIGURATION_ROOT.rglob('*.yaml') if 'queues' not in path.parts)
QUEUE_FILES = sorted((CONFIGURATION_ROOT / 'queues').glob('*.yaml'))


def _configuration_id(path: Path) -> str:
    return path.relative_to(CONFIGURATION_ROOT).as_posix()


@pytest.mark.parametrize('path', EXPERIMENT_FILES, ids=_configuration_id)
def test_every_experiment_configuration_resolves(path: Path) -> None:
    load_experiment_configuration(path)


@pytest.mark.parametrize('path', QUEUE_FILES, ids=_configuration_id)
def test_every_queue_configuration_loads(path: Path) -> None:
    load_queue_configuration(path)
