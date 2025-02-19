# Test the remove_repetitions function
from src.util.remove_repetitions import remove_repetitions


if __name__ == '__main__':

    def test(i: list[int], o: list[int]) -> None:
        res_indices = remove_repetitions(i)
        res = [i[j] for j in res_indices]
        assert res == o, f'Expected {o}, got {res}'

    base = [1, 2, 3, 4]
    repeated = base * 3

    print('Testing remove_repetitions')
    test([], [])
    test(repeated, base)
    test(base * 5, base)
    test(base * 6, base + base)
    test([4] + repeated + [3], [4] + base + [3])

    print('All tests passed')
