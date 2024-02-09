from chess import Square


def set_square(bitboard: int, square: Square) -> int:
    """Set the given square on the bitboard to 1."""
    return bitboard | (1 << square)


def clear_square(bitboard: int, square: Square) -> int:
    """Clear the given square on the bitboard to 0."""
    return bitboard & ~(1 << square)


def toggle_square(bitboard: int, square: Square) -> int:
    """Toggle the given square on the bitboard between 0 and 1."""
    return bitboard ^ (1 << square)


def square_is_set(bitboard: int, square: Square) -> bool:
    """Returns true if the given square is set to 1 on the bitboard, otherwise false."""
    return ((bitboard >> square) & 1) != 0


def get_index_of_LSB(bitboard: int) -> int:
    """Returns index of the first bit that is set to 1."""
    lsb = bitboard & -bitboard
    return (lsb - 1).bit_length()


def clear_and_get_index_of_LSB(bitboard: int) -> tuple[int, int]:
    """Returns index of the first bit that is set to 1. The bit will also be cleared to zero."""
    lsb = bitboard & -bitboard
    index = (lsb - 1).bit_length()
    new_bitboard = bitboard & ~lsb
    return index, new_bitboard


def get_number_of_set_bits(bitboard: int) -> int:
    """Returns the number of bits that are set to 1 in the given bitboard."""
    return bin(bitboard).count('1')
