import json
import struct
from typing import Tuple, List

from src.dataset.SelfPlayDatasetStats import SelfPlayDatasetStats

SelfPlaySample = Tuple[List[int], List[Tuple[int, int]], float]


def load_selfplay_file(filename: str, load_samples: bool = True) -> Tuple[SelfPlayDatasetStats, List[SelfPlaySample]]:
    with open(filename, 'rb') as f:

        def read_uint32() -> int:
            return struct.unpack('<I', f.read(4))[0]

        # Read and verify magic number.
        magic = f.read(4)
        if magic != b'SMPF':
            raise ValueError('Invalid file format')

        # Read version.
        version = read_uint32()
        if version != 1:
            raise ValueError(f'Unsupported version: {version}')

        # Read metadata.
        metadata_length = read_uint32()
        metadata_json = f.read(metadata_length).decode('utf-8')
        metadata_dict = json.loads(metadata_json)
        stats = SelfPlayDatasetStats(**metadata_dict)

        assert 0 <= stats.num_games <= 100000, f'Invalid number of games: {stats.num_games}'
        assert 0 <= stats.num_samples <= 10000000, f'Invalid number of samples: {stats.num_samples}'

        if not load_samples:
            return stats, []

        # Read sample count.
        sample_count = read_uint32()
        samples: List[SelfPlaySample] = []
        for _ in range(sample_count):
            # Read board: 14 64-bit unsigned ints.
            board = []
            for _ in range(14):
                board.append(struct.unpack('<Q', f.read(8))[0])
            # Read the number of visitCount pairs.
            num_pairs = read_uint32()
            visit_counts = []
            for _ in range(num_pairs):
                move = read_uint32()
                count = read_uint32()
                assert 0 <= move < 2000, f'Invalid move: {move}'
                assert 0 < count < 1000000, f'Invalid visit count: {count}'
                visit_counts.append((move, count))
            # Read result score (32-bit float).
            result_score = struct.unpack('<f', f.read(4))[0]
            samples.append((board, visit_counts, result_score))
    return stats, samples


def write_selfplay_file(filename: str, stats: SelfPlayDatasetStats, samples: List[SelfPlaySample]) -> None:
    with open(filename, 'wb') as f:

        def write_uint32(value: int):
            f.write(struct.pack('<I', value))

        # Write magic number and version.
        f.write(b'SMPF')
        write_uint32(1)

        # Write metadata as JSON.
        metadata_json = json.dumps(stats._asdict()).encode('utf-8')
        assert 0 <= stats.num_games <= 100000, f'Invalid number of games: {stats.num_games}'
        assert 0 <= stats.num_samples <= 10000000, f'Invalid number of samples: {stats.num_samples}'
        write_uint32(len(metadata_json))
        f.write(metadata_json)

        # Write sample count.
        write_uint32(len(samples))
        for board, visit_counts, result_score in samples:
            assert len(board) == 14, f'Expected 14 elements in board, got {len(board)}'
            # Write board as 14 64-bit unsigned ints.
            for element in board:
                assert 0 <= element < 2**64, f'Invalid board element: {element}'
                f.write(struct.pack('<Q', element))
            # Write visitCounts.
            write_uint32(len(visit_counts))
            for move, count in visit_counts:
                assert 0 <= move < 2000, f'Invalid move: {move}'
                assert 0 < count < 1000000, f'Invalid visit count: {count}'
                write_uint32(move)
                write_uint32(count)
            # Write result score (32-bit float).
            f.write(struct.pack('<f', result_score))
