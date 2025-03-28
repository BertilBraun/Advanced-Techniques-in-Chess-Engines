import json
import struct
from dataclasses import dataclass


@dataclass
class Stats:
    num_samples: int
    num_games: int
    game_lengths: int
    total_generation_time: float
    resignations: int
    num_too_long_games: int


@dataclass
class Sample:
    board: list[int]
    visitCounts: list[tuple]
    resultScore: float


def load_selfplay_file(filename: str) -> tuple[Stats, list[Sample]]:
    with open(filename, 'rb') as f:
        # Read the magic number (4 bytes) and check it.
        magic = f.read(4)
        if magic != b'SMPF':
            raise ValueError('Invalid file format')

        # Read version (uint32).
        version_bytes = f.read(4)
        version = struct.unpack('I', version_bytes)[0]
        if version != 1:
            raise ValueError(f'Unsupported version: {version}')

        # Read metadata JSON length (uint32).
        metadata_length_bytes = f.read(4)
        metadata_length = struct.unpack('I', metadata_length_bytes)[0]

        # Read metadata JSON string.
        metadata_json = f.read(metadata_length).decode('utf-8')
        metadata_dict = json.loads(metadata_json)

        # Extract stats from metadata.
        stats = Stats(
            num_samples=metadata_dict.get('num_samples', 0),
            num_games=metadata_dict.get('num_games', 0),
            game_lengths=metadata_dict.get('game_lengths', 0),
            total_generation_time=metadata_dict.get('total_generation_time', 0.0),
            resignations=metadata_dict.get('resignations', 0),
            num_too_long_games=metadata_dict.get('num_too_long_games', 0),
        )

        # Read sample count.
        sample_count_bytes = f.read(4)
        sample_count = struct.unpack('I', sample_count_bytes)[0]

        samples = []
        for _ in range(sample_count):
            # Read board: 14 64-bit unsigned ints.
            board_bytes = f.read(14 * 8)
            board = list(struct.unpack('14Q', board_bytes))

            # Read number of visitCount pairs (uint32).
            num_pairs_bytes = f.read(4)
            num_pairs = struct.unpack('I', num_pairs_bytes)[0]

            pairs = []
            for _ in range(num_pairs):
                # Each pair: two 32-bit integers.
                pair_bytes = f.read(8)
                first, second = struct.unpack('ii', pair_bytes)
                pairs.append((first, second))

            # Read result score (32-bit float).
            result_score_bytes = f.read(4)
            result_score = struct.unpack('f', result_score_bytes)[0]

            samples.append(Sample(board=board, visitCounts=pairs, resultScore=result_score))

    return stats, samples


# Example usage:
if __name__ == '__main__':
    filename = 'your_prefix_0.bin'
    stats, samples = load_selfplay_file(filename)
    print('Stats:')
    print(stats)
    print('Number of samples:', len(samples))
