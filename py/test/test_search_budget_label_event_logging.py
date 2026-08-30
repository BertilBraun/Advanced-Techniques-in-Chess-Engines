from __future__ import annotations

import re
from pathlib import Path

from src.search_budget.manager import (
    FailedLabelJobReport,
    GenerationLabelReport,
    SkippedLabelJobReport,
)

_REPORTING_SOURCES = (
    'src/training/coordinator.py',
    'src/training/reporting.py',
    'src/training/search_budget_tensorboard.py',
)


def _known_attribute_names() -> set[str]:
    names: set[str] = set()
    for model in (GenerationLabelReport, FailedLabelJobReport, SkippedLabelJobReport):
        names |= set(model.model_fields)
        names |= {name for name in dir(model) if not name.startswith('_')}
    return names


def test_label_event_reporting_only_reads_attributes_the_reports_carry() -> None:
    # A stale attribute in one of these paths only raises when a label job is collected, which is
    # long after startup and after the first training quanta have already been credited.
    known = _known_attribute_names()
    missing: list[str] = []
    for relative_path in _REPORTING_SOURCES:
        source = Path(relative_path).read_text(encoding='utf-8')
        for attribute in re.findall(r'\bevent\.([a-z_][a-z0-9_]*)', source):
            if attribute not in known:
                missing.append(f'{relative_path}: event.{attribute}')
    assert not missing, f'label-event reporting reads attributes no report carries: {missing}'
