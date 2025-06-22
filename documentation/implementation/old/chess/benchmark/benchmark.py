import chess
import time

def generate_moves_benchmark(board, depth=3):
    if depth < 1:
        return
    for move in board.legal_moves:
        board_copy = board.copy(stack=False)
        board_copy.push(move)
        generate_moves_benchmark(board_copy, depth - 1)

def generate_moves_benchmark_main(iterations, depth):
    total_duration = 0

    for _ in range(iterations):
        board = chess.Board()
        start = time.time()

        generate_moves_benchmark(board, depth)

        end = time.time()
        total_duration += end - start

    average_time = total_duration / iterations
    print(f"Average time for generating moves and pushing/popping a move: {average_time} seconds.")
    print(f"Total time for generating moves and pushing/popping a move: {total_duration} seconds.")

def copy_state_benchmark(iterations):
    board = chess.Board()
    total_duration = 0

    for _ in range(iterations):
        board_copy= None
        start = time.time()

        board_copy = board.copy(stack=False)

        end = time.time()
        total_duration += end - start
        if board_copy is None:
            pass

    average_time = total_duration / iterations
    print(f"Average time for copying board state: {average_time} seconds.")
    print(f"Total time for copying board state: {total_duration} seconds.")

if __name__ == "__main__":
    # generate_moves_benchmark_main(100, 4)
    copy_state_benchmark(100_000_000)
