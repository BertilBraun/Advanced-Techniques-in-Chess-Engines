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
    visitCounts: list[tuple[int, int]]
    resultScore: float


def load_selfplay_stats(filename: str) -> Stats:
    return load_selfplay_file(filename, load_samples=False)[0]


def load_selfplay_file(filename: str, load_samples: bool = True) -> tuple[Stats, list[Sample]]:
    with open(filename, 'rb') as f:

        def read_uint32() -> int:
            # Read 4 bytes and unpack as unsigned int (little-endian).
            return struct.unpack('I', f.read(4))[0]

        # Read the magic number (4 bytes) and check it.
        magic = f.read(4)
        if magic != b'SMPF':
            raise ValueError('Invalid file format')

        # Read version (uint32).
        version = read_uint32()
        if version != 1:
            raise ValueError(f'Unsupported version: {version}')

        # Read metadata JSON length (uint32).
        metadata_length = read_uint32()

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

        if not load_samples:
            return stats, []

        # Read sample count.
        sample_count = read_uint32()

        samples = []
        for _ in range(sample_count):
            # Read board: 14 64-bit unsigned ints.
            board = list(struct.unpack('14Q', f.read(14 * 8)))

            # Read number of visitCount pairs (uint32).
            num_pairs = read_uint32()

            visit_counts = []
            for _ in range(num_pairs):
                # Each pair: two uint32
                move, count = read_uint32(), read_uint32()
                visit_counts.append((move, count))

            # Read result score (32-bit float).
            result_score = struct.unpack('f', f.read(4))[0]

            samples.append(Sample(board=board, visitCounts=visit_counts, resultScore=result_score))

    return stats, samples


def write_selfplay_file(filename: str, stats: Stats, samples: list[Sample]):
    with open(filename, 'wb') as f:

        def write_uint32(value: int):
            # Pack unsigned int (little-endian) and write 4 bytes.
            f.write(struct.pack('I', value))

        assert f.write(b'SMPF') == 4  # Magic number

        write_uint32(1)  # Version

        metadata_json = json.dumps(stats.__dict__).encode('utf-8')
        write_uint32(len(metadata_json))
        assert f.write(metadata_json) == len(metadata_json)

        write_uint32(len(samples))
        for sample in samples:
            f.write(struct.pack('14Q', *sample.board))  # 14 64-bit unsigned ints

            write_uint32(len(sample.visitCounts))  # Number of visitCount pairs
            for move, count in sample.visitCounts:
                write_uint32(move)
                write_uint32(count)

            f.write(struct.pack('f', sample.resultScore))  # Result score (32-bit float)


# Example usage:
if __name__ == '__main__':
    filename = 'your_prefix_0.bin'
    stats, samples = load_selfplay_file(filename)
    print('Stats:')
    print(stats)
    print('Number of samples:', len(samples))
