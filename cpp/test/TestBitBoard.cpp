#include "util/BitBoard.hpp"
#include "util/PackedPlane.hpp"

#include <array>
#include <cstdint>
#include <vector>

template <std::size_t BoardSize> bool runBitBoardChecks() {
    using Board = BitBoard<BoardSize>;

    constexpr typename Board::Point firstPoint{0, 0};
    constexpr typename Board::Point lastPoint{
        static_cast<std::uint8_t>(BoardSize - 1),
        static_cast<std::uint8_t>(BoardSize - 1),
    };

    static_assert(Board::index(firstPoint) == 0);
    static_assert(Board::point(0) == firstPoint);
    static_assert(Board::point(Board::bit_count - 1) == lastPoint);

    const Board emptyBoard;
    if (emptyBoard.any() || !emptyBoard.none() || emptyBoard.count() != 0) {
        return false;
    }

    const Board fullBoard = Board::full();
    if (!fullBoard.any() || fullBoard.count() != Board::bit_count) {
        return false;
    }
    if constexpr (Board::bit_count % Board::word_bits != 0) {
        if (fullBoard.word(Board::word_count - 1) >> (Board::bit_count % Board::word_bits)) {
            return false;
        }
    }

    Board diagonal;
    diagonal.set(firstPoint);
    diagonal.set(lastPoint);
    if (!diagonal.test(firstPoint) || !diagonal.test(lastPoint) || diagonal.count() != 2) {
        return false;
    }

    Board antiDiagonal = Board::from_point(lastPoint);
    antiDiagonal.set(Board::point(BoardSize - 1));
    if (!antiDiagonal.intersects(diagonal)) {
        return false;
    }

    const Board unionBoard = diagonal | antiDiagonal;
    if (!unionBoard.contains(diagonal) || unionBoard.count() != 3) {
        return false;
    }

    const Board difference = unionBoard - diagonal;
    if (difference.count() != 1 || !difference.test(Board::point(BoardSize - 1))) {
        return false;
    }

    std::vector<typename Board::Point> iteratedPoints;
    for (const typename Board::Point point : unionBoard.set_bits()) {
        iteratedPoints.push_back(point);
    }
    if (iteratedPoints.size() != 3) {
        return false;
    }

    Board popped = unionBoard;
    typename Board::Point poppedPoint{};
    std::size_t poppedCount = 0;
    while (popped.pop_first(poppedPoint)) {
        ++poppedCount;
    }
    if (poppedCount != 3 || popped.any()) {
        return false;
    }

    constexpr std::size_t binaryPlaneCount = 2;
    std::array<Board, binaryPlaneCount> planes{};
    planes[0] = diagonal;
    planes[1] = ~Board::from_point(firstPoint);
    std::vector<std::int8_t> payload(PackedPlaneLayout<BoardSize, binaryPlaneCount, 0>::binary_bytes);
    serialize_binary_planes<BoardSize, binaryPlaneCount>(planes, payload);
    const auto decodedPlanes =
        deserialize_binary_planes<BoardSize, binaryPlaneCount>(payload);
    if (decodedPlanes != planes) {
        return false;
    }

    return true;
}

int main() {
    return runBitBoardChecks<7>() && runBitBoardChecks<8>() && runBitBoardChecks<9>() ? 0 : 1;
}
