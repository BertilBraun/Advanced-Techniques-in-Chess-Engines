import json
import struct
from typing import Tuple, List

from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats

SelfPlaySample = Tuple[List[int], List[Tuple[int, int]], float]


def load_selfplay_file(filename: str, load_samples: bool = True) -> Tuple[SelfPlayDatasetStats, List[SelfPlaySample]]:
    with open(filename, 'rb') as f:

        def read_uint32() -> int:
            return struct.unpack('I', f.read(4))[0]

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

        if not load_samples:
            return stats, []

        # Read sample count.
        sample_count = read_uint32()
        samples: List[SelfPlaySample] = []
        for _ in range(sample_count):
            # Read board: 14 64-bit unsigned ints.
            board = list(struct.unpack('14Q', f.read(14 * 8)))
            # Read the number of visitCount pairs.
            num_pairs = read_uint32()
            visit_counts = []
            for _ in range(num_pairs):
                move = read_uint32()
                count = read_uint32()
                visit_counts.append((move, count))
            # Read result score (32-bit float).
            result_score = struct.unpack('f', f.read(4))[0]
            samples.append((board, visit_counts, result_score))
    return stats, samples


def write_selfplay_file(filename: str, stats: SelfPlayDatasetStats, samples: List[SelfPlaySample]) -> None:
    with open(filename, 'wb') as f:

        def write_uint32(value: int):
            f.write(struct.pack('I', value))

        # Write magic number and version.
        f.write(b'SMPF')
        write_uint32(1)

        # Write metadata as JSON.
        metadata_json = json.dumps(stats._asdict()).encode('utf-8')
        write_uint32(len(metadata_json))
        f.write(metadata_json)

        # Write sample count.
        write_uint32(len(samples))
        for board, visit_counts, result_score in samples:
            if len(board) != 14:
                raise ValueError('Board must have 14 numbers')
            # Write board as 14 64-bit unsigned ints.
            f.write(struct.pack('14Q', *board))
            # Write visitCounts.
            write_uint32(len(visit_counts))
            for move, count in visit_counts:
                write_uint32(move)
                write_uint32(count)
            # Write result score (32-bit float).
            f.write(struct.pack('f', result_score))
