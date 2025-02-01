from typing import Any


def remove_repetitions(items: list[Any]) -> list[int]:
    """
    moves: a list of items to remove repetitions from
    Returns a new list of indices which should be keept where any sub-block of length >= 4 that occurs >= 3 times in a row is collapsed into a single occurrence of that block.
    """
    filtered_moves: list[int] = []
    i = 0
    n = len(items)

    while i < n:
        found_repeat = False
        # The longest block we can repeat 3x from position i is at most (n - i) // 3
        max_len = (n - i) // 3
        # Try from the largest possible block down to 4
        for length in range(max_len, 3, -1):
            block = items[i : i + length]
            # Check if block repeats 3 times consecutively
            if items[i + length : i + 2 * length] == block and items[i + 2 * length : i + 3 * length] == block:
                # Count how many times it actually repeats
                times = 3
                while i + times * length < n and items[i + times * length : i + (times + 1) * length] == block:
                    times += 1
                # Keep exactly one occurrence of that block
                filtered_moves.extend(list(range(i, i + length)))
                # Skip over all the repeated blocks
                i += times * length
                found_repeat = True
                break

        # If no repeated block found at position i, just take one move
        if not found_repeat:
            filtered_moves.append(i)
            i += 1

    return filtered_moves
