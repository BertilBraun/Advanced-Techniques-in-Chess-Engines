from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from src.evaluation.configuration import EvaluationSearchConfiguration
from src.evaluation.match import SearchActionSelector
from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.self_play.configuration import BatchedInferenceParams
from src.training.checkpoint.contracts import CheckpointReference


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--position-count", type=int, default=50)
    parser.add_argument("--searches", type=int, default=80_000)
    parser.add_argument("--warmup-searches", type=int, default=1_024)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    loaded = load_experiment_configuration(arguments.experiment)
    if not isinstance(loaded, ChessExperimentConfiguration):
        raise ValueError("The timing benchmark requires a chess experiment.")
    checkpoint = CheckpointReference.load_for_inference(arguments.run_directory, arguments.generation)
    game = ChessImplementation(loaded)
    opening_payload = json.loads(arguments.openings.read_text(encoding="utf-8"))
    opening_records = opening_payload["openings"][: arguments.position_count]
    positions = []
    for opening in opening_records:
        position = game.state.initial_position()
        for action_id in opening["action_ids"]:
            position = game.state.child_position(position, action_id)
        positions.append(position)
    position_batch = tuple(positions)
    if len(position_batch) != arguments.position_count:
        raise ValueError("The opening manifest does not contain enough positions.")

    search_configuration = EvaluationSearchConfiguration(
        searches_per_move=arguments.searches,
        parallel_searches=8,
        exploration_constant=1.0,
        inference=BatchedInferenceParams(
            inference_workers=1,
            inference_batch_size=64,
            outstanding_batches_per_worker=1,
        ),
    )
    search = game.create_evaluation_search(arguments.device, checkpoint, search_configuration)
    selector = SearchActionSelector(search, arguments.searches, search_configuration.parallel_searches)
    warmup_selector = SearchActionSelector(search, arguments.warmup_searches, search_configuration.parallel_searches)
    warmup_selector.choose_actions(position_batch)

    started = time.perf_counter()
    selected_actions = selector.choose_actions(position_batch)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "device_id": arguments.device,
        "source_revision": "92c9f487d38467ac8c34bb88a28604e3a539e15f",
        "experiment_configuration_sha256": experiment_configuration_sha256(loaded),
        "checkpoint_generation": checkpoint.generation,
        "inference_model_sha256": file_sha256(checkpoint.inference_model_path),
        "opening_manifest_sha256": file_sha256(arguments.openings),
        "position_count": len(position_batch),
        "selected_action_count": len(selected_actions),
        "searches_per_position": arguments.searches,
        "parallel_searches": 8,
        "inference_workers": 1,
        "inference_batch_size": 64,
        "outstanding_batches_per_worker": 1,
        "warmup_searches_per_position": arguments.warmup_searches,
        "elapsed_seconds": elapsed_seconds,
        "amortized_seconds_per_position": elapsed_seconds / len(position_batch),
        "searches_per_second": arguments.searches * len(position_batch) / elapsed_seconds,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
