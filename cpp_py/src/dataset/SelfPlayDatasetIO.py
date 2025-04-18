import json
from typing import Tuple, List

from src.dataset.SelfPlayDatasetStats import SelfPlayDatasetStats

SelfPlaySample = Tuple[List[int], List[Tuple[int, int]], float]


STATS_FILE_POSTFIX = '_stats.json'
BOARDS_FILE_POSTFIX = '_boards.csv'
MOVES_FILE_POSTFIX = '_moves.csv'


def load_selfplay_file(filename: str, load_samples: bool = True) -> Tuple[SelfPlayDatasetStats, List[SelfPlaySample]]:
    # === Read Stats JSON from {filename}_stats.json ===
    stats_filename = filename + STATS_FILE_POSTFIX
    with open(stats_filename, 'r') as f:
        stats = SelfPlayDatasetStats(**json.load(f))

    if not load_samples:
        return stats, []

    # === Read Boards as hex CSV from {filename}_boards.csv ===
    boards_filename = filename + BOARDS_FILE_POSTFIX
    with open(boards_filename, 'r') as f:
        lines = f.readlines()
        samples: List[SelfPlaySample] = []
        for line in lines:
            # Split the line by comma and convert each hex string to an integer.
            parts = line.strip().split(',')
            board = [int(part, 16) for part in parts[:-1]]
            result_score = float(parts[-1])
            samples.append((board, [], result_score))

    # === Read Visit Counts (Moves) from {filename}_moves.csv ===
    moves_filename = filename + MOVES_FILE_POSTFIX
    with open(moves_filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            # Skip the header line
            parts = line.strip().split(',')
            sample_index = int(parts[0])
            move = int(parts[1])
            count = int(parts[2])

            # Append the visit count to the corresponding sample
            assert 0 <= sample_index < len(samples), f'Sample index {sample_index} out of range for samples list.'
            assert 0 <= move < 2000, f'Move {move} out of range.'
            assert 0 <= count < 65536, f'Count {count} out of range.'
            samples[sample_index][1].append((move, count))

    return stats, samples


def write_selfplay_file(base_filename: str, stats: SelfPlayDatasetStats, samples: List[SelfPlaySample]) -> None:
    # === Write Stats JSON to {base_filename}_stats.json ===
    stats_filename = base_filename + STATS_FILE_POSTFIX
    with open(stats_filename, 'w') as f:
        # Write pretty JSON (4 space indent)
        json.dump(
            {
                'num_samples': stats.num_samples,
                'num_games': stats.num_games,
                'game_lengths': stats.game_lengths,
                'total_generation_time': stats.total_generation_time,
                'resignations': stats.resignations,
                'num_too_long_games': stats.num_too_long_games,
            },
            f,
            indent=4,
        )

    # === Write Boards as hex CSV to {base_filename}_boards.csv ===
    boards_filename = base_filename + BOARDS_FILE_POSTFIX
    with open(boards_filename, 'w') as f:
        for board, _, result_score in samples:
            # Format each board element as a 16-character, zero-padded hexadecimal string.
            board_hex = ','.join(format(num, '016x') for num in board)
            # Append the result score at the end.
            line = f'{board_hex},{result_score}\n'
            f.write(line)

    # === Write Visit Counts (Moves) to {base_filename}_moves.csv ===
    moves_filename = base_filename + MOVES_FILE_POSTFIX
    with open(moves_filename, 'w') as f:
        # Optionally write a CSV header.
        f.write('sample_index,move,count\n')
        for sample_index, (_, visit_counts, _) in enumerate(samples):
            for move, count in visit_counts:
                f.write(f'{sample_index},{move},{count}\n')
