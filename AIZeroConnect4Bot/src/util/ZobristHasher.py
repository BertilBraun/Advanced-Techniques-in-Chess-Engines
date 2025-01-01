import torch


class ZobristHasher:
    def __init__(self, planes: int, rows: int, cols: int):
        self.planes = planes
        self.rows = rows
        self.cols = cols
        self.size = planes * rows * cols
        self.keys_per_device: dict[torch.device, torch.Tensor] = {}

    def _create_zobrist_keys(self, device: torch.device) -> torch.Tensor:
        """Create random zobrist keys on a Device."""
        return torch.randint(
            low=-(2**63),
            high=2**63 - 1,
            size=(self.planes, self.rows, self.cols),
            dtype=torch.long,
            device=device,
        ).view(self.size)

    def _get_keys_for_device(self, device: torch.device) -> torch.Tensor:
        """Return zobrist keys on the specified device, creating/caching if necessary."""
        if device not in self.keys_per_device:
            self.keys_per_device[device] = self._create_zobrist_keys(device)
        return self.keys_per_device[device]

    def zobrist_hash_boards(self, boards: torch.Tensor) -> list[int]:
        assert boards.shape[1:] == (self.planes, self.rows, self.cols), f'Invalid shape: {boards.shape}'
        keys = self._get_keys_for_device(boards.device)
        return _zobrist_hash_boards(boards, keys).tolist()


def _xor_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.transpose(dim, -1).contiguous()
    D = x.size(-1)

    # Find largest power of 2 less than or equal to D
    largest_pow2 = 1
    while largest_pow2 <= D:
        largest_pow2 <<= 1
    largest_pow2 >>= 1

    # XOR reduce largest_pow2 elements
    length = largest_pow2
    y = x[..., :largest_pow2]  # Take the first largest_pow2 elements
    while length > 1:
        y = y.view(-1, length // 2, 2)
        y = y[:, :, 0] ^ y[:, :, 1]
        length //= 2
    y = y.view(*x.shape[:-1])

    # XOR the remainder
    remainder = D - largest_pow2
    if remainder > 0:
        # XOR each remaining element
        for i in range(largest_pow2, D):
            y = y ^ x[..., i]

    return y


def _zobrist_hash_boards(boards: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    N = boards.size(0)
    boards = boards.to(torch.int64)
    boards_flat = boards.view(N, keys.size(0))
    keys_expanded = keys.unsqueeze(0).expand(N, keys.size(0))
    selected_keys = keys_expanded * boards_flat
    hashes = _xor_reduce(selected_keys, dim=1)
    return hashes


if torch.cuda.is_available():
    _zobrist_hash_boards = torch.compile(_zobrist_hash_boards)
