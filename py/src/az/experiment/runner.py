from __future__ import annotations

from src.az.experiment.evaluation_phase import run_evaluation
from src.az.experiment.lifecycle import (
    ExperimentPhase,
    ExperimentRunRepository,
    ExperimentRunState,
    ExperimentStatus,
)
from src.az.experiment.reporting_phase import run_reporting
from src.az.experiment.training_phase import run_training_window

__all__ = [
    'run_evaluation',
    'run_remaining',
    'run_reporting',
    'run_training_window',
]


def run_remaining(repository: ExperimentRunRepository) -> ExperimentRunState:
    while True:
        state = repository.load()
        if state.status is ExperimentStatus.COMPLETE or state.stop_requested:
            return state
        match state.next_phase:
            case ExperimentPhase.TRAINING_RUN:
                run_training_window(repository)
            case ExperimentPhase.EVALUATION:
                run_evaluation(repository)
            case ExperimentPhase.REPORTING:
                run_reporting(repository)
            case ExperimentPhase.COMPLETE:
                return state
