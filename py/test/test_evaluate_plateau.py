from pathlib import Path

from src.experiment.plateau import PlateauDecision, PlateauStatus
from tools.evaluate_plateau import PlateauInput, write_plateau_decision


def test_write_plateau_decision_is_machine_readable(tmp_path: Path) -> None:
    decision_path = tmp_path / 'plateau-decision.json'
    decision = PlateauDecision(
        status=PlateauStatus.PLATEAU,
        reason='No meaningful gain.',
        evaluated_iterations=(5, 10, 15),
    )

    write_plateau_decision(decision_path, decision)

    loaded = PlateauDecision.model_validate_json(decision_path.read_text(encoding='utf-8'))
    assert loaded == decision
    assert PlateauInput.model_fields.keys() == {'rule', 'evaluations'}
