import torch
import random
from typing import Iterable, TypeVar


T = TypeVar('T')


def lerp(a: T, b: T, t: float) -> T:
    return t * a + (1 - t) * b  # type: ignore


def random_id() -> str:
    random_base = 1_000_000_000
    return str(random.randint(random_base, random_base * 10))


def batched_iterate(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_zobrist_table(device) -> torch.Tensor:
    from AIZeroConnect4Bot.src.settings import CELL_STATES, COLUMN_COUNT, ROW_COUNT

    # Generate random 64-bit keys for each cell and state
    total_cells = ROW_COUNT * COLUMN_COUNT
    zobrist_table = torch.randint(
        low=-(2**63),
        high=2**63 - 1,
        size=(total_cells, CELL_STATES),
        dtype=torch.long,
        device=device,
    )
    return zobrist_table


ZOBRIST_TABLES: dict[torch.device, torch.Tensor] = {}


def zobrist_hash_boards(boards: torch.Tensor) -> list[int]:
    # boards shape: (batch, ROW_COUNT, COL_COUNT)
    # zobrist_table shape: (ROW_COUNT*COL_COUNT, 3)

    # Get the zobrist table for the device
    device = boards.device
    if device not in ZOBRIST_TABLES:
        ZOBRIST_TABLES[device] = create_zobrist_table(device)
    zobrist_table = ZOBRIST_TABLES[device]

    boards = boards.squeeze(1)

    batch, row_count, col_count = boards.shape
    total_cells = row_count * col_count
    assert zobrist_table.shape == (total_cells, 3), "Zobrist table shape doesn't match board dimensions"

    # Map states -1->2, 0->0, 1->1
    mapped_states = torch.where(
        boards == -1,
        torch.tensor(2, device=boards.device),
        torch.where(
            boards == 1,
            torch.tensor(1, device=boards.device),
            torch.tensor(0, device=boards.device),
        ),
    )

    # Flatten (batch, ROW_COUNT*COL_COUNT)
    mapped_states = mapped_states.view(batch, total_cells)

    # Create a cell index tensor
    cell_indices = torch.arange(total_cells, device=boards.device).unsqueeze(0).expand(batch, total_cells)

    # Fetch keys
    hash_values = zobrist_table[cell_indices, mapped_states]

    # XOR reduce along cell dimension to get final hashes
    batch_hash = hash_values[:, 0]
    for i in range(1, total_cells):
        batch_hash ^= hash_values[:, i]
    return batch_hash.tolist()


def hash_board(state: torch.Tensor) -> int:
    assert len(state.shape) >= 2, f'State must have at least 2 dimensions: {state.shape}'
    # TODO must be a better way to do this - faster while still managing hash collisions

    # Map -1 to 2, 0 to 0, 1 to 1
    mapped_state = torch.where(state == -1, 2, torch.where(state == 1, 1, 0)).to(torch.uint8)
    # Flatten the last two dimensions
    mapped_state = mapped_state.reshape(-1)
    # Pad to multiple of 4
    pad_length = (-len(mapped_state)) % 4
    if pad_length > 0:
        mapped_state = torch.nn.functional.pad(mapped_state, (0, pad_length), value=0)
    # Reshape and pack 2-bit values into bytes
    mapped_state = mapped_state.view(-1, 4)
    bytes_array = (mapped_state[:, 0] << 6) | (mapped_state[:, 1] << 4) | (mapped_state[:, 2] << 2) | mapped_state[:, 3]
    # Convert to integer
    h = int.from_bytes(bytes_array.cpu().numpy().tobytes(), byteorder='big')
    return h


def hash_boards(states: torch.Tensor) -> list[int]:
    assert len(states.shape) == 4, f'States must have 4 dimensions: {states.shape}'
    states = states.squeeze(1)

    # Map -1 to 2, 0 to 0, 1 to 1
    mapped_states = torch.where(states == -1, 2, states).to(torch.uint8)

    # Flatten the last two dimensions
    batch_size = mapped_states.shape[0]
    mapped_states = mapped_states.view(batch_size, -1)

    # Pad to multiple of 4
    pad_length = (-mapped_states.size(1)) % 4
    if pad_length > 0:
        mapped_states = torch.nn.functional.pad(mapped_states, (0, pad_length), value=0)

    # Reshape and pack 2-bit values into bytes
    mapped_states = mapped_states.view(batch_size, -1, 4)
    bytes_array = (
        (mapped_states[:, :, 0] << 6)
        | (mapped_states[:, :, 1] << 4)
        | (mapped_states[:, :, 2] << 2)
        | (mapped_states[:, :, 3] << 0)
    )

    # Convert to integer
    hashes = []
    for i in range(batch_size):
        h = int.from_bytes(bytes_array[i].cpu().numpy().tobytes(), byteorder='big')
        hashes.append(h)
    return hashes


def hash_boards_simple(states: torch.Tensor) -> list[int]:
    hashes = []
    for i in range(states.shape[0]):
        hash = 0
        for v in (states[i] + 1).view(-1):
            hash = 3 * hash + v.item()
        hashes.append(hash)
    return hashes


# time compare hash_board, and zobrist_hash_boards
if __name__ == '__main__':
    import timeit

    state = torch.randint(-1, 2, (128, 1, 8, 7), dtype=torch.int8)
    print(state)
    print(hash_boards(state))
    print(zobrist_hash_boards(state))

    print('Time hash_board:', timeit.timeit(lambda: hash_boards(state), number=1000))
    # print('Time hash_board_simple:', timeit.timeit(lambda: hash_boards_simple(state), number=1000))
    # TODO zobrist is ultra slow
    print('Time zobrist:', timeit.timeit(lambda: zobrist_hash_boards(state), number=1000))
