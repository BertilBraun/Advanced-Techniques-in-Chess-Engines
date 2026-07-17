import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.experiment.plateau import (
    CheckpointEvaluation,
    PlateauDecision,
    PlateauRule,
    evaluate_plateau,
)


class PlateauInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    rule: PlateauRule
    evaluations: tuple[CheckpointEvaluation, ...]


class CommandLineArguments(argparse.Namespace):
    input_path: Path
    output_path: Path


def parse_arguments() -> CommandLineArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input_path', type=Path, required=True)
    parser.add_argument('--output', dest='output_path', type=Path, required=True)
    return parser.parse_args(namespace=CommandLineArguments())


def write_plateau_decision(path: Path, decision: PlateauDecision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.tmp')
    temporary_path.write_text(
        decision.model_dump_json(indent=2) + '\n',
        encoding='utf-8',
    )
    temporary_path.replace(path)


def main() -> None:
    arguments = parse_arguments()
    plateau_input = PlateauInput.model_validate_json(arguments.input_path.read_text(encoding='utf-8'))
    decision = evaluate_plateau(plateau_input.evaluations, plateau_input.rule)
    write_plateau_decision(arguments.output_path, decision)


if __name__ == '__main__':
    main()
