import argparse
import hashlib
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.experiment.run_configuration import RunConfiguration, load_run_configuration
from src.games.chess.ChessSettings import ensure_eval_dataset_exists


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = 'https://database.nikonoel.fr/lichess_elite_2024-10.zip'


class ChessEvaluationDatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    dataset_path: str
    dataset_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_url: str
    source_year: int
    source_month: int
    source_game_count: int
    random_seed: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-config', required=True, type=Path)
    parser.add_argument('--manifest-output', required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(
    configuration: RunConfiguration,
    manifest_output: Path,
) -> ChessEvaluationDatasetManifest:
    configured_path = configuration.evaluation_protocol.evaluation_dataset_path
    if configured_path is None:
        raise ValueError('Run configuration does not enable a fixed evaluation dataset.')
    dataset_path = SOURCE_ROOT / configured_path

    random.seed(configuration.workload.random_seed)
    ensure_eval_dataset_exists(str(dataset_path))
    if not dataset_path.is_file():
        raise ValueError(f'Evaluation dataset was not created: {dataset_path}')

    manifest = ChessEvaluationDatasetManifest(
        dataset_path=configured_path,
        dataset_sha256=file_sha256(dataset_path),
        source_url=SOURCE_URL,
        source_year=2024,
        source_month=10,
        source_game_count=50,
        random_seed=configuration.workload.random_seed,
    )
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_output.with_suffix('.json.tmp')
    temporary_path.write_text(manifest.model_dump_json(indent=2) + '\n', encoding='utf-8')
    temporary_path.replace(manifest_output)
    return manifest


def main() -> None:
    arguments = parse_arguments()
    configuration = load_run_configuration(arguments.run_config)
    manifest = prepare_dataset(configuration, arguments.manifest_output)
    print(manifest.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
