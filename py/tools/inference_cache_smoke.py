import argparse
import json
from pathlib import Path

from AlphaZeroCpp import InferenceClientParams, MCTS, MCTSParams, new_root


STARTING_POSITION = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--cache-capacity', type=int, default=64)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    client_params = InferenceClientParams(
        device_id=0,
        currentModelPath=str(arguments.model_path),
        maxBatchSize=16,
        microsecondsTimeoutInferenceThread=500,
        cacheCapacity=arguments.cache_capacity,
    )
    mcts_params = MCTSParams(
        num_parallel_searches=1,
        num_full_searches=8,
        num_fast_searches=8,
        c_param=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        min_visit_count=0,
        num_threads=1,
    )
    search = MCTS(client_params, mcts_params)
    results = search.search([(new_root(STARTING_POSITION), False)])
    statistics, _ = search.get_inference_statistics()

    if len(results.results) != 1:
        raise RuntimeError('Expected exactly one MCTS result')
    if statistics.cacheCapacity != arguments.cache_capacity:
        raise RuntimeError('Inference cache did not retain its configured capacity')
    if statistics.cacheFingerprintCollisions != 0:
        raise RuntimeError('Inference cache reported an unexpected board-fingerprint collision')

    print(
        json.dumps(
            {
                'capacity': statistics.cacheCapacity,
                'entries': statistics.uniquePositions,
                'evictions': statistics.cacheEvictions,
                'fingerprint_collisions': statistics.cacheFingerprintCollisions,
            },
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
