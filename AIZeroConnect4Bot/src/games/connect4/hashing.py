import torch

from AIZeroConnect4Bot.src.games.connect4.Connect4Defines import ROW_COUNT, COLUMN_COUNT, CELL_STATES


def create_zobrist_table(device) -> torch.Tensor:
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


# @torch.compile
def _zobrist_hash_boards(boards: torch.Tensor, zobrist_table: torch.Tensor) -> torch.Tensor:
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
    return batch_hash


def zobrist_hash_boards(boards: torch.Tensor) -> list[int]:
    # boards shape: (batch, ROW_COUNT, COL_COUNT)
    # zobrist_table shape: (ROW_COUNT*COL_COUNT, 3)

    # Get the zobrist table for the device
    device = boards.device
    if device not in ZOBRIST_TABLES:
        ZOBRIST_TABLES[device] = create_zobrist_table(device)
    zobrist_table = ZOBRIST_TABLES[device]

    return _zobrist_hash_boards(boards, zobrist_table).tolist()
