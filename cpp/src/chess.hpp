#pragma once

#include "commonBase.hpp"

namespace chess {
enum Color : bool { WHITE, BLACK };
inline constexpr Color operator!(Color color) {
    return color == Color::WHITE ? Color::BLACK : Color::WHITE;
}

inline constexpr std::array<Color, 2> COLORS = {Color::WHITE, Color::BLACK};
inline const std::array<std::string, 2> COLOR_NAMES = {"black", "white"};

enum PieceType { NONE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NUM_PIECE_TYPES };
inline constexpr std::array<PieceType, 6> PIECE_TYPES = {PieceType::PAWN,   PieceType::KNIGHT,
                                                         PieceType::BISHOP, PieceType::ROOK,
                                                         PieceType::QUEEN,  PieceType::KING};
inline constexpr std::array<PieceType, 7> PIECE_TYPES_AND_NONE = {
    PieceType::NONE, PieceType::PAWN,  PieceType::KNIGHT, PieceType::BISHOP,
    PieceType::ROOK, PieceType::QUEEN, PieceType::KING};

inline constexpr int operator+(int a, PieceType b) { return a + static_cast<int>(b); }

inline const std::array<std::pair<int, int>, 8> KNIGHT_MOVES = {
    {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}}};
inline const std::array<std::pair<int, int>, 4> ROOK_MOVES = {{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}};
inline const std::array<std::pair<int, int>, 4> BISHOP_MOVES = {
    {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};

inline const std::array<char, PieceType::NUM_PIECE_TYPES> PIECE_SYMBOLS = {' ', 'p', 'n', 'b',
                                                                           'r', 'q', 'k'};
inline const std::array<std::string, PieceType::NUM_PIECE_TYPES> PIECE_NAMES = {
    "none", "pawn", "knight", "bishop", "rook", "queen", "king"};

inline constexpr bool isValidPieceSymbol(char c) {
    return c == 'p' || c == 'n' || c == 'b' || c == 'r' || c == 'q' || c == 'k' || c == 'P' ||
           c == 'N' || c == 'B' || c == 'R' || c == 'Q' || c == 'K';
}
inline char pieceSymbol(PieceType piece_type) {
    return PIECE_SYMBOLS[static_cast<int>(piece_type)];
}

inline std::string pieceName(PieceType piece_type) {
    return PIECE_NAMES[static_cast<int>(piece_type)];
}
inline const std::unordered_map<char, std::string> UNICODE_PIECE_SYMBOLS = {
    {'R', "♖"}, {'r', "♜"}, {'N', "♘"}, {'n', "♞"}, {'B', "♗"}, {'b', "♝"},
    {'Q', "♕"}, {'q', "♛"}, {'K', "♔"}, {'k', "♚"}, {'P', "♙"}, {'p', "♟"},
};
inline const std::array<std::string, 8> FILE_NAMES = {"a", "b", "c", "d", "e", "f", "g", "h"};
inline const std::array<std::string, 8> RANK_NAMES = {"1", "2", "3", "4", "5", "6", "7", "8"};

enum class Status : int {
    VALID = 0,
    NO_WHITE_KING = 1 << 0,
    NO_BLACK_KING = 1 << 1,
    TOO_MANY_KINGS = 1 << 2,
    TOO_MANY_WHITE_PAWNS = 1 << 3,
    TOO_MANY_BLACK_PAWNS = 1 << 4,
    PAWNS_ON_BACK_RANK = 1 << 5,
    TOO_MANY_WHITE_PIECES = 1 << 6,
    TOO_MANY_BLACK_PIECES = 1 << 7,
    BAD_CASTLING_RIGHTS = 1 << 8,
    INVALID_EP_SQUARE = 1 << 9,
    OPPOSITE_CHECK = 1 << 10,
    EMPTY = 1 << 11,
    RACE_CHECK = 1 << 12,
    RACE_OVER = 1 << 13,
    RACE_MATERIAL = 1 << 14,
    TOO_MANY_CHECKERS = 1 << 15,
    IMPOSSIBLE_CHECK = 1 << 16,
};

inline constexpr Status operator|(Status a, Status b) {
    return (Status) (static_cast<int>(a) | static_cast<int>(b));
}
inline constexpr Status operator|=(Status &a, Status b) {
    a = a | b;
    return a;
}
inline constexpr Status operator&(Status a, Status b) {
    return (Status) (static_cast<int>(a) & static_cast<int>(b));
}
inline constexpr Status operator&=(Status &a, Status b) {
    a = a & b;
    return a;
}

enum class Termination {
    CHECKMATE,
    // See :func:`chess::Board.is_checkmate()`.
    STALEMATE,
    // See :func:`chess::Board.is_stalemate()`.
    INSUFFICIENT_MATERIAL,
    // See :func:`chess::Board.is_insufficient_material()`.
    SEVENTYFIVE_MOVES,
};

class Outcome {
public:
    Termination termination;
    std::optional<Color> winner;

    Outcome(Termination termination, std::optional<Color> winner = std::nullopt)
        : termination(termination), winner(winner) {}

    std::string result() const {
        if (!winner.has_value()) {
            return "1/2-1/2";
        }
        return winner == Color::WHITE ? "1-0" : "0-1";
    }
};

// clang-format off
enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
};

inline constexpr int operator+(Square square) { return static_cast<int>(square); }
inline constexpr int operator-(Square square) { return -static_cast<int>(square); }
inline constexpr Square operator+(Square square, int n) { return static_cast<Square>(static_cast<int>(square) + n); }
inline constexpr Square operator-(Square square, int n) { return static_cast<Square>(static_cast<int>(square) - n); }
inline constexpr Square &operator+=(Square &square, int n) { return square = square + n; }
inline constexpr Square &operator-=(Square &square, int n) { return square = square - n; }

inline constexpr std::array<Square, 64> SQUARES = {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
};

inline const std::array<std::string, 64> SQUARE_NAMES = {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
};
// clang-format on

inline Square parseSquare(const std::string &name) {
    for (Square sq : SQUARES) {
        if (SQUARE_NAMES[sq] == name) {
            return sq;
        }
    }
    throw std::invalid_argument("Invalid square name");
}

inline std::string squareName(Square square) {
    if (square < 0 || square >= 64) {
        throw std::invalid_argument("Square index out of bounds");
    }
    return SQUARE_NAMES[square];
}

inline constexpr Square square(int file_index, int rank_index) {
    return (Square) (rank_index * 8 + file_index);
}

inline constexpr Square squareFile(Square square) { return (Square) (square & 7); }

inline constexpr Square squareRank(Square square) { return (Square) (square >> 3); }

inline constexpr Square squareFlipVertical(Square square) {
    // https://www.chessprogramming.org/Vertical_Flipping
    return (Square) (square ^ 56);
}

inline constexpr Square squareFlipHorizontal(Square square) {
    // https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#:~:text=return%20_byteswap_uint64(x)%3B%0A%7D-,Horizontal,-.%201%201%201
    return (Square) (square ^ 7);
}

inline constexpr std::array<Square, 64> __createSquares180() {
    std::array<Square, 64> squares_180{};
    for (size_t i = 0; i < SQUARES.size(); ++i) {
        squares_180[i] = squareFlipVertical(SQUARES[i]);
    }
    return squares_180;
}

inline constexpr std::array<Square, 64> SQUARES_180 = __createSquares180();

using Bitboard = unsigned long long;
inline constexpr Bitboard BB_EMPTY = 0;
inline constexpr Bitboard BB_ALL = 0xFFFF'FFFF'FFFF'FFFF;

inline constexpr std::array<Bitboard, 64> __createBBSquares() {
    std::array<Bitboard, 64> squares{};
    for (int i = 0; i < 64; ++i) {
        squares[i] = 1ULL << i;
    }
    return squares;
}

inline constexpr std::array<Bitboard, 64> BB_SQUARES = __createBBSquares();

// clang-format off
enum BB_SQUARE : Bitboard {
    BB_A1 = BB_SQUARES[A1], BB_B1 = BB_SQUARES[B1], BB_C1 = BB_SQUARES[C1], BB_D1 = BB_SQUARES[D1], BB_E1 = BB_SQUARES[E1], BB_F1 = BB_SQUARES[F1], BB_G1 = BB_SQUARES[G1], BB_H1 = BB_SQUARES[H1],
    BB_A2 = BB_SQUARES[A2], BB_B2 = BB_SQUARES[B2], BB_C2 = BB_SQUARES[C2], BB_D2 = BB_SQUARES[D2], BB_E2 = BB_SQUARES[E2], BB_F2 = BB_SQUARES[F2], BB_G2 = BB_SQUARES[G2], BB_H2 = BB_SQUARES[H2],
    BB_A3 = BB_SQUARES[A3], BB_B3 = BB_SQUARES[B3], BB_C3 = BB_SQUARES[C3], BB_D3 = BB_SQUARES[D3], BB_E3 = BB_SQUARES[E3], BB_F3 = BB_SQUARES[F3], BB_G3 = BB_SQUARES[G3], BB_H3 = BB_SQUARES[H3],
    BB_A4 = BB_SQUARES[A4], BB_B4 = BB_SQUARES[B4], BB_C4 = BB_SQUARES[C4], BB_D4 = BB_SQUARES[D4], BB_E4 = BB_SQUARES[E4], BB_F4 = BB_SQUARES[F4], BB_G4 = BB_SQUARES[G4], BB_H4 = BB_SQUARES[H4],
    BB_A5 = BB_SQUARES[A5], BB_B5 = BB_SQUARES[B5], BB_C5 = BB_SQUARES[C5], BB_D5 = BB_SQUARES[D5], BB_E5 = BB_SQUARES[E5], BB_F5 = BB_SQUARES[F5], BB_G5 = BB_SQUARES[G5], BB_H5 = BB_SQUARES[H5],
    BB_A6 = BB_SQUARES[A6], BB_B6 = BB_SQUARES[B6], BB_C6 = BB_SQUARES[C6], BB_D6 = BB_SQUARES[D6], BB_E6 = BB_SQUARES[E6], BB_F6 = BB_SQUARES[F6], BB_G6 = BB_SQUARES[G6], BB_H6 = BB_SQUARES[H6],
    BB_A7 = BB_SQUARES[A7], BB_B7 = BB_SQUARES[B7], BB_C7 = BB_SQUARES[C7], BB_D7 = BB_SQUARES[D7], BB_E7 = BB_SQUARES[E7], BB_F7 = BB_SQUARES[F7], BB_G7 = BB_SQUARES[G7], BB_H7 = BB_SQUARES[H7],
    BB_A8 = BB_SQUARES[A8], BB_B8 = BB_SQUARES[B8], BB_C8 = BB_SQUARES[C8], BB_D8 = BB_SQUARES[D8], BB_E8 = BB_SQUARES[E8], BB_F8 = BB_SQUARES[F8], BB_G8 = BB_SQUARES[G8], BB_H8 = BB_SQUARES[H8],
};
// clang-format on

inline constexpr Bitboard BB_CORNERS = BB_A1 | BB_H1 | BB_A8 | BB_H8;
inline constexpr Bitboard BB_CENTER = BB_D4 | BB_E4 | BB_D5 | BB_E5;

inline constexpr Bitboard BB_LIGHT_SQUARES = 0x55AA'55AA'55AA'55AA;
inline constexpr Bitboard BB_DARK_SQUARES = 0xAA55'AA55'AA55'AA55;

inline constexpr std::array<Bitboard, 8> __createBBFiles() {
    std::array<Bitboard, 8> files{};
    for (int i = 0; i < 8; ++i) {
        files[i] = 0x0101'0101'0101'0101ULL << i;
    }
    return files;
}

inline constexpr std::array<Bitboard, 8> BB_FILES = __createBBFiles();

// clang-format off
enum BB_FILE : Bitboard {
    BB_FILE_A = BB_FILES[0],
    BB_FILE_B = BB_FILES[1],
    BB_FILE_C = BB_FILES[2],
    BB_FILE_D = BB_FILES[3],
    BB_FILE_E = BB_FILES[4],
    BB_FILE_F = BB_FILES[5],
    BB_FILE_G = BB_FILES[6],
    BB_FILE_H = BB_FILES[7],
};
// clang-format on

inline constexpr std::array<Bitboard, 8> __createBBRanks() {
    std::array<Bitboard, 8> ranks{};
    for (int i = 0; i < 8; ++i) {
        ranks[i] = 0xFFull << (8 * i);
    }
    return ranks;
}

inline constexpr std::array<Bitboard, 8> BB_RANKS = __createBBRanks();

// clang-format off
enum BB_RANK : Bitboard {
    BB_RANK_1 = BB_RANKS[0],
    BB_RANK_2 = BB_RANKS[1],
    BB_RANK_3 = BB_RANKS[2],
    BB_RANK_4 = BB_RANKS[3],
    BB_RANK_5 = BB_RANKS[4],
    BB_RANK_6 = BB_RANKS[5],
    BB_RANK_7 = BB_RANKS[6],
    BB_RANK_8 = BB_RANKS[7],
};
// clang-format on

inline constexpr Bitboard BB_BACK_RANKS = BB_RANK_1 | BB_RANK_8;

inline int squareDistance(Square a, Square b) {
    return std::max(std::abs(squareFile(a) - squareFile(b)),
                    std::abs(squareRank(a) - squareRank(b)));
}

inline int squareManhattanDistance(Square a, Square b) {
    return std::abs(squareFile(a) - squareFile(b)) + std::abs(squareRank(a) - squareRank(b));
}

inline int squareKnightDistance(Square a, Square b) {
    int dx = std::abs(squareFile(a) - squareFile(b));
    int dy = std::abs(squareRank(a) - squareRank(b));

    if (dx + dy == 1) {
        return 3;
    } else if (dx == 2 && dy == 2) {
        return 4;
    } else if (dx == 1 && dy == 1) {
        // Special case only for corner squares
        if ((BB_SQUARES[a] & BB_CORNERS) || (BB_SQUARES[b] & BB_CORNERS))
            return 4;
    }

    int m = (int) std::ceil(std::max({dx / 2.0, dy / 2.0, (dx + dy) / 3.0}));
    return m + ((m + dx + dy) % 2);
}

inline constexpr int lsb(Bitboard bb) {
    if (bb == 0)
        return -1;
    int n = 0;
    if (!(bb & 0xFFFFFFFF)) {
        bb >>= 32;
        n += 32;
    }
    if (!(bb & 0xFFFF)) {
        bb >>= 16;
        n += 16;
    }
    if (!(bb & 0xFF)) {
        bb >>= 8;
        n += 8;
    }
    if (!(bb & 0xF)) {
        bb >>= 4;
        n += 4;
    }
    if (!(bb & 0x3)) {
        bb >>= 2;
        n += 2;
    }
    if (!(bb & 0x1)) {
        n += 1;
    }
    return n;
}

inline constexpr int msb(Bitboard bb) {
    if (bb == 0)
        return -1;
    int n = 0;
    if (bb & 0xFFFFFFFF00000000) {
        bb >>= 32;
        n += 32;
    }
    if (bb & 0xFFFF0000) {
        bb >>= 16;
        n += 16;
    }
    if (bb & 0xFF00) {
        bb >>= 8;
        n += 8;
    }
    if (bb & 0xF0) {
        bb >>= 4;
        n += 4;
    }
    if (bb & 0xC) {
        bb >>= 2;
        n += 2;
    }
    if (bb & 0x2) {
        n += 1;
    }
    return n;
}

inline std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

inline constexpr int popcount(Bitboard bb) {
    const uint64_t m1 = 0x5555555555555555;  // binary: 0101...
    const uint64_t m2 = 0x3333333333333333;  // binary: 00110011..
    const uint64_t m4 = 0x0f0f0f0f0f0f0f0f;  // binary:  4 zeros,  4 ones ...
    const uint64_t h01 = 0x0101010101010101; // the sum of 256 to the power of 0,1,2,3...

    bb -= (bb >> 1) & m1;              // put count of each 2 bits into those 2 bits
    bb = (bb & m2) + ((bb >> 2) & m2); // put count of each 4 bits into those 4 bits
    bb = (bb + (bb >> 4)) & m4;        // put count of each 8 bits into those 8 bits
    return (bb * h01) >> 56; // returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}

inline std::vector<Square> scanForward(Bitboard bb) {
    std::vector<Square> squares;
    squares.reserve(popcount(bb));
    while (bb) {
        int sq = lsb(bb);
        squares.push_back((Square) sq);
        bb &= bb - 1; // Reset LSB
    }
    return squares;
}

inline std::vector<Square> scanReversed(Bitboard bb) {
    std::vector<Square> squares;
    squares.reserve(popcount(bb));
    while (bb) {
        int sq = msb(bb);
        squares.push_back((Square) sq);
        bb &= ~(1ULL << sq); // Reset MSB
    }
    return squares;
}

inline constexpr Bitboard flipVertical(Bitboard bb) {
    bb = ((bb >> 8) & 0x00FF'00FF'00FF'00FF) | ((bb & 0x00FF'00FF'00FF'00FF) << 8);
    bb = ((bb >> 16) & 0x0000'FFFF'0000'FFFF) | ((bb & 0x0000'FFFF'0000'FFFF) << 16);
    bb = (bb >> 32) | ((bb & 0x0000'0000'FFFF'FFFF) << 32);
    return bb;
}

inline constexpr Bitboard flipHorizontal(Bitboard bb) {
    bb = ((bb >> 1) & 0x5555'5555'5555'5555) | ((bb & 0x5555'5555'5555'5555) << 1);
    bb = ((bb >> 2) & 0x3333'3333'3333'3333) | ((bb & 0x3333'3333'3333'3333) << 2);
    bb = ((bb >> 4) & 0x0F0F'0F0F'0F0F'0F0F) | ((bb & 0x0F0F'0F0F'0F0F'0F0F) << 4);
    return bb;
}

inline constexpr Bitboard flipDiagonal(Bitboard bb) {
    Bitboard t = 0;
    t = (bb ^ (bb << 28)) & 0x0F0F'0F0F'0000'0000ULL;
    bb = bb ^ t ^ (t >> 28);
    t = (bb ^ (bb << 14)) & 0x3333'0000'3333'0000ULL;
    bb = bb ^ t ^ (t >> 14);
    t = (bb ^ (bb << 7)) & 0x5500'5500'5500'5500ULL;
    bb = bb ^ t ^ (t >> 7);
    return bb;
}

inline constexpr Bitboard flipAntiDiagonal(Bitboard bb) {
    Bitboard t = 0;
    t = bb ^ (bb << 36);
    bb = bb ^ ((t ^ (bb >> 36)) & 0xF0F0'F0F0'0F0F'0F0FULL);
    t = (bb ^ (bb << 18)) & 0xCCCC'0000'CCCC'0000ULL;
    bb = bb ^ t ^ (t >> 18);
    t = (bb ^ (bb << 9)) & 0xAA00'AA00'AA00'AA00ULL;
    bb = bb ^ t ^ (t >> 9);
    return bb;
}

inline constexpr Bitboard shiftDown(Bitboard b) { return b >> 8; }

inline constexpr Bitboard shiftUp(Bitboard b) { return (b << 8) & BB_ALL; }

inline Bitboard _slidingAttacks(Square square, Bitboard occupied, const std::vector<int> &deltas) {
    Bitboard attacks = BB_EMPTY;

    for (auto delta : deltas) {
        Square sq = square;

        while (true) {
            sq += delta;
            if (!(0 <= sq && sq < 64) || squareDistance(sq, sq - delta) > 2) {
                break;
            }

            attacks |= BB_SQUARES[sq];

            if (occupied & BB_SQUARES[sq]) {
                break;
            }
        }
    }

    return attacks;
}

inline std::array<Bitboard, 64> BB_KNIGHT_ATTACKS = [] {
    std::array<Bitboard, 64> attacks{};
    for (Square sq : SQUARES) {
        attacks[sq] = _slidingAttacks(sq, BB_ALL, {17, 15, 10, 6, -17, -15, -10, -6});
    }
    return attacks;
}();

inline std::array<Bitboard, 64> BB_KING_ATTACKS = [] {
    std::array<Bitboard, 64> attacks{};
    for (Square sq : SQUARES) {
        attacks[sq] = _slidingAttacks(sq, BB_ALL, {9, 8, 7, 1, -9, -8, -7, -1});
    }
    return attacks;
}();

inline std::array<std::array<Bitboard, 64>, 2> BB_PAWN_ATTACKS = [] {
    std::array<std::array<Bitboard, 64>, 2> attacks{};
    for (int color = 0; color < 2; ++color) {
        for (Square sq : SQUARES) {
            attacks[color][sq] = _slidingAttacks(
                sq, BB_ALL, color == 0 ? std::vector<int>{-7, -9} : std::vector<int>{7, 9});
        }
    }
    return attacks;
}();

inline constexpr Bitboard _edges(Square square) {
    return (((BB_RANKS[0] | BB_RANKS[7]) & ~BB_RANKS[squareRank(square)]) |
            ((BB_FILES[0] | BB_FILES[7]) & ~BB_FILES[squareFile(square)]));
}

inline std::vector<Bitboard> _carryRippler(Bitboard mask) {
    std::vector<Bitboard> subsets;
    subsets.reserve(popcount(mask) + 1);
    Bitboard subset = BB_EMPTY;
    do {
        subsets.push_back(subset);
        subset = (subset - mask) & mask;
    } while (subset != BB_EMPTY);
    return subsets;
}

inline std::array<Bitboard, 64> _mask_table(const std::vector<int> &deltas) {
    std::array<Bitboard, 64> mask_table{};
    for (Square square : SQUARES) {
        mask_table[square] = _slidingAttacks(square, 0, deltas) & ~_edges(square);
    }
    return mask_table;
}

inline std::array<std::unordered_map<Bitboard, Bitboard>, 64>
_attack_table(const std::vector<int> &deltas) {
    std::array<std::unordered_map<Bitboard, Bitboard>, 64> attack_table;
    for (Square square : SQUARES) {
        std::unordered_map<Bitboard, Bitboard> attacks;
        Bitboard mask = _slidingAttacks(square, 0, deltas) & ~_edges(square);
        for (auto subset : _carryRippler(mask)) {
            attacks[subset] = _slidingAttacks(square, subset, deltas);
        }
        attack_table[square] = attacks;
    }
    return attack_table;
}

inline auto BB_DIAG_MASKS = _mask_table({-9, -7, 7, 9});
inline auto BB_FILE_MASKS = _mask_table({-8, 8});
inline auto BB_RANK_MASKS = _mask_table({-1, 1});

inline auto BB_DIAG_ATTACKS = _attack_table({-9, -7, 7, 9});
inline auto BB_FILE_ATTACKS = _attack_table({-8, 8});
inline auto BB_RANK_ATTACKS = _attack_table({-1, 1});

inline std::array<std::array<Bitboard, 64>, 64> BB_RAYS = [] {
    std::array<std::array<Bitboard, 64>, 64> rays{};
    for (Square a : SQUARES) {
        for (Square b : SQUARES) {
            if (BB_DIAG_ATTACKS[a][0] & BB_SQUARES[b]) {
                rays[a][b] = ((BB_DIAG_ATTACKS[a][0] & BB_DIAG_ATTACKS[b][0]) | BB_SQUARES[a] |
                              BB_SQUARES[b]);
            } else if (BB_RANK_ATTACKS[a][0] & BB_SQUARES[b]) {
                rays[a][b] = (BB_RANK_ATTACKS[a][0] | BB_SQUARES[a]);
            } else if (BB_FILE_ATTACKS[a][0] & BB_SQUARES[b]) {
                rays[a][b] = (BB_FILE_ATTACKS[a][0] | BB_SQUARES[a]);
            } else {
                rays[a][b] = (BB_EMPTY);
            }
        }
    }
    return rays;
}();

inline Bitboard ray(int a, int b) { return BB_RAYS[a][b]; }

inline Bitboard between(int a, int b) {
    Bitboard bb = BB_RAYS[a][b] & ((BB_ALL << a) ^ (BB_ALL << b));
    return bb & (bb - 1);
}

class Piece {
public:
    // Constructors
    Piece() : m_value(0) {}

    Piece(PieceType piece_type, Color color)
        : m_value(((static_cast<unsigned char>(piece_type) & 0b111) << 1) |
                  (static_cast<unsigned char>(color) & 1)) {}

    Piece(const Piece &other) : m_value(other.m_value) {}

    Piece &operator=(const Piece &other) {
        m_value = other.m_value;
        return *this;
    }

    // Get the piece type
    PieceType pieceType() const { return static_cast<PieceType>(m_value >> 1); }

    // Get the piece color
    Color color() const { return static_cast<Color>(m_value & 1); }

    // Get the symbol of the piece
    char symbol() const {
        auto symbol = pieceSymbol(pieceType());
        return color() == Color::WHITE ? (char) toupper(symbol) : (char) tolower(symbol);
    }

    // Get the Unicode symbol of the piece
    std::string unicode_symbol(bool invert_color = false) const {
        char symbolKey = symbol();
        if (invert_color) {
            symbolKey = isupper(symbolKey) ? (char) tolower(symbolKey) : (char) toupper(symbolKey);
        }
        return UNICODE_PIECE_SYMBOLS.at(symbolKey);
    }

    // Overload hash function
    size_t hash() const { return m_value; }

    // Class method to create Piece from symbol
    static Piece from_symbol(char symbol) {
        for (PieceType pieceType : PIECE_TYPES) {
            if (tolower(pieceSymbol(pieceType)) == tolower(symbol)) {
                return Piece(pieceType, isupper(symbol) ? Color::WHITE : Color::BLACK);
            }
        }
        throw std::invalid_argument("Invalid symbol for piece creation");
    }

private:
    // Encoded as: 0bPPPC where PPP = PieceType (3 bits), C = Color (1 bit)
    unsigned char m_value;
};

class Move {
public:
    Move() : m_value(0) {}

    Move(int from_square, int to_square, PieceType promotion = PieceType::NONE)
        : m_value(getValue(from_square, to_square, promotion)) {}

    Move(const Move &other) : m_value(other.m_value) {}

    Move &operator=(const Move &other) {
        m_value = other.m_value;
        return *this;
    }

    Square fromSquare() const {
        return (Square) ((m_value & FROM_SQUARE_MASK) >> FROM_SQUARE_SHIFT);
    }

    Square toSquare() const { return (Square) ((m_value & TO_SQUARE_MASK) >> TO_SQUARE_SHIFT); }

    PieceType promotion() const {
        return static_cast<PieceType>((m_value & PROMOTION_MASK) >> PROMOTION_SHIFT);
    }

    std::string uci() const {
        if (!*this)
            return "0000";

        std::string uci = SQUARE_NAMES[fromSquare()] + SQUARE_NAMES[toSquare()];
        auto promo = promotion();
        if (promo != PieceType::NONE) {
            uci += pieceSymbol(promo);
        }
        return uci;
    }

    bool operator!() const { return m_value == 0; }

    bool operator==(const Move &other) const { return m_value == other.m_value; }

    static Move null() { return Move(); }

    static Move fromUci(const std::string &uci) {
        if (uci == "0000") {
            return null();
        }
        if (uci.size() != 4 && uci.size() != 5) {
            throw std::invalid_argument("Invalid uci string");
        }
        int from = parseSquare(uci.substr(0, 2));
        int to = parseSquare(uci.substr(2, 2));
        PieceType promotion = PieceType::NONE;
        if (uci.size() == 5) {
            promotion = Piece::from_symbol(uci[4]).pieceType();
        }
        return Move(from, to, promotion);
    }

private:
    // 1111110000000000 for from_square (6 bits shifted left by 10)
    static constexpr unsigned short FROM_SQUARE_MASK = 0xFC00;
    static constexpr int FROM_SQUARE_SHIFT = 10;

    // 0000001111110000 for to_square (6 bits shifted left by 4)
    static constexpr unsigned short TO_SQUARE_MASK = 0x03F0;
    static constexpr int TO_SQUARE_SHIFT = 4;

    // 0000000000001111 for promotion (4 bits)
    static constexpr unsigned short PROMOTION_MASK = 0x000F;
    static constexpr int PROMOTION_SHIFT = 0;

    unsigned short getValue(int from_square, int to_square, PieceType promotion) {
        unsigned short value = 0;
        value |= (static_cast<unsigned short>(from_square) << FROM_SQUARE_SHIFT);
        value |= (static_cast<unsigned short>(to_square) << TO_SQUARE_SHIFT);
        value |= (static_cast<unsigned short>(promotion) << PROMOTION_SHIFT);
        return value;
    }

    unsigned short m_value;
};

class SquareSet {
private:
    Bitboard mask;

public:
    // Constructor from a Bitboard mask or individual squares
    explicit SquareSet(Bitboard mask = BB_EMPTY) : mask(mask & BB_ALL) {}

    // Checks if a square is in the set
    bool contains(Square square) const { return (BB_SQUARES[square] & mask) != 0; }

    // Adds a square to the set
    void add(Square square) { mask |= BB_SQUARES[square]; }

    // Discards a square from the set
    void discard(Square square) { mask &= ~BB_SQUARES[square]; }

    // Returns the number of squares in the set
    size_t size() const { return popcount(mask); }

    // Checks if the set is empty
    bool empty() const { return mask == 0; }

    // Set operations
    SquareSet unionWith(const SquareSet &other) const { return SquareSet(mask | other.mask); }

    SquareSet intersection(const SquareSet &other) const { return SquareSet(mask & other.mask); }

    SquareSet difference(const SquareSet &other) const { return SquareSet(mask & ~other.mask); }

    SquareSet symmetricDifference(const SquareSet &other) const {
        return SquareSet(mask ^ other.mask);
    }

    Square pop() {
        Square square = (Square) lsb(mask);
        mask &= mask - 1;
        return square;
    }

    // Clear the set
    void clear() { mask = BB_EMPTY; }

    // Iteration over squares in the set
    std::vector<Square> squares() const { return scanForward(mask); }

    // Printing
    std::string toString() const {
        std::string result = "";
        for (Square square : SQUARES_180) {
            result += (contains(square) ? "1" : ".");
            if (squareRank(square) != 7) {
                result += " ";
            }
            if (squareFile(square) == 7) {
                result += "\n";
            }
        }
        return result;
    }

    // Access underlying bitboard
    Bitboard toBitboard() const { return mask; }

    // Overloaded operators for set operations
    SquareSet operator|(const SquareSet &other) const { return unionWith(other); }
    SquareSet operator&(const SquareSet &other) const { return intersection(other); }
    SquareSet operator-(const SquareSet &other) const { return difference(other); }
    SquareSet operator^(const SquareSet &other) const { return symmetricDifference(other); }
    SquareSet &operator|=(const SquareSet &other) {
        mask |= other.mask;
        return *this;
    }
    SquareSet &operator&=(const SquareSet &other) {
        mask &= other.mask;
        return *this;
    }
    SquareSet &operator-=(const SquareSet &other) {
        mask &= ~other.mask;
        return *this;
    }
    SquareSet &operator^=(const SquareSet &other) {
        mask ^= other.mask;
        return *this;
    }

    // Conversion to bool (non-explicit) checks for non-emptiness
    operator bool() const { return !empty(); }

    // Equality
    bool operator==(const SquareSet &other) const { return mask == other.mask; }
    bool operator!=(const SquareSet &other) const { return mask != other.mask; }

    // Output stream operator for easy printing
    friend std::ostream &operator<<(std::ostream &os, const SquareSet &ss) {
        os << ss.toString();
        return os;
    }
};

class Board {
public:
    Color turn;
    // The side to move (``chess::WHITE`` or ``chess::BLACK``).

    Bitboard castling_rights;
    // Bitmask of the rooks with castling rights.
    //
    // To test for specific squares:
    //
    // >>> import chess
    // >>>
    // >>> board = chess::Board()
    // >>> bool(board.castling_rights & chess::BB_H1)  # White can castle with the h1 rook
    // True
    //
    // To add a specific square:
    //
    // >>> board.castling_rights |= chess::BB_A1
    //
    // Use :func:`~chess::Board.set_castling_fen()` to set multiple castling
    // rights. Also see :func:`~chess::Board.has_castling_rights()`,
    // :func:`~chess::Board.has_kingside_castling_rights()`,
    // :func:`~chess::Board.has_queenside_castling_rights()`,
    // :func:`~chess::Board.clean_castling_rights()`.

    std::optional<Square> ep_square;
    // The potential en passant square on the third or sixth rank or ``None``.
    //
    // Use :func:`~chess::Board.has_legal_en_passant()` to test if en passant
    // capturing would actually be possible on the next move.

    int fullmove_number;
    // Counts move pairs. Starts at `1` and is incremented after every move of the black side.

    int halfmove_clock;
    // The number of half-moves since the last capture or pawn move.
private:
    std::array<Bitboard, 2> m_occupied_color; // Bitboards for white and black pieces
    Bitboard m_pawns, m_knights, m_bishops, m_rooks, m_queens, m_kings, m_promoted, m_occupied;

    std::optional<std::vector<Move>> m_cachedLegalMoves = std::nullopt;
    std::optional<std::vector<Move>> m_cachedPseudoLegalMoves = std::nullopt;
    std::optional<std::vector<Move>> m_cachedLegalEPMoves = std::nullopt;
    std::optional<std::vector<Move>> m_cachedPseudoLegalEPMoves = std::nullopt;

public:
    static Board fromFEN(const std::string &fen) {
        auto parts = split(fen, ' ');

        if (parts.size() != 6 && parts.size() != 4) {
            throw std::invalid_argument("expected 6/4 parts in fen: " + fen);
        }

        std::string board_fen = parts[0];

        std::vector<std::string> rows = split(board_fen, '/');
        if (rows.size() != 8) {
            throw std::invalid_argument("expected 8 rows in position part of fen: " + board_fen);
        }

        for (const auto &row : rows) {
            int field_sum = 0;
            bool previous_was_digit = false;
            bool previous_was_piece = false;

            for (char c : row) {
                if (isdigit(c)) {
                    if (previous_was_digit) {
                        throw std::invalid_argument(
                            "two subsequent digits in position part of fen: " + board_fen);
                    }
                    field_sum += c - '0';
                    previous_was_digit = true;
                    previous_was_piece = false;
                } else if (c == '~') {
                    if (!previous_was_piece) {
                        throw std::invalid_argument(
                            "'~' not after piece in position part of fen: " + board_fen);
                    }
                    previous_was_digit = false;
                    previous_was_piece = false;
                } else if (isValidPieceSymbol(c)) {
                    field_sum += 1;
                    previous_was_digit = false;
                    previous_was_piece = true;
                } else {
                    throw std::invalid_argument("invalid character in position part of fen: " +
                                                board_fen);
                }
            }

            if (field_sum != 8) {
                throw std::invalid_argument("expected 8 columns per row in position part of fen: " +
                                            board_fen);
            }
        }

        Board board(false); // Create an empty board

        int squareIndex = 0;
        for (char c : board_fen) {
            if (isdigit(c)) {
                squareIndex += c - '0';
            } else if (isValidPieceSymbol(c)) {
                Piece piece = Piece::from_symbol(c);
                board._setPieceAt(SQUARES_180[squareIndex], piece.pieceType(), piece.color());
                squareIndex += 1;
            } else if (c == '~') {
                board.m_promoted |= BB_SQUARES[SQUARES_180[squareIndex - 1]];
            }
        }

        board.turn = parts[1] == "w" ? WHITE : BLACK;

        board.castling_rights = 0;

        for (char c : parts[2]) {
            switch (c) {
            case 'K':
                board.castling_rights |= chess::BB_H1;
                break;
            case 'Q':
                board.castling_rights |= chess::BB_A1;
                break;
            case 'k':
                board.castling_rights |= chess::BB_H8;
                break;
            case 'q':
                board.castling_rights |= chess::BB_A8;
                break;
            case '-':
                break; // No castling rights
            default:
                throw std::invalid_argument("Invalid castling rights in FEN.");
            }
        }

        if (parts[3] == "-") {
            board.ep_square = std::nullopt;
        } else {
            board.ep_square = parseSquare(parts[3]);
        }

        if (parts.size() > 4) {
            board.halfmove_clock = std::stoi(parts[4]);
            board.fullmove_number = std::stoi(parts[5]);
        } else {
            board.halfmove_clock = 0;
            board.fullmove_number = 1;
        }

        return board;
    }

    Board(bool starting_position = true)
        : turn(WHITE), castling_rights(BB_CORNERS), ep_square(std::nullopt), fullmove_number(1),
          halfmove_clock(0) {
        if (starting_position) {
            resetBoard();
        } else {
            clearBoard();
        }
    }

    // A dynamic list of legal moves.
    std::vector<Move> legalMoves() { return _generateLegalMoves(); }

    // A dynamic list of pseudo-legal moves, much like the legal move list.
    //
    // Pseudo-legal moves might leave or put the king in check, but are
    // otherwise valid. Null moves are not pseudo-legal. Castling moves are
    // only included if they are completely legal.
    std::vector<Move> pseudoLegalMoves() { return _generatePseudoLegalMoves(); }

    void resetBoard() {
        // Resets pieces to the starting position.

        m_pawns = BB_RANK_2 | BB_RANK_7;
        m_knights = BB_B1 | BB_G1 | BB_B8 | BB_G8;
        m_bishops = BB_C1 | BB_F1 | BB_C8 | BB_F8;
        m_rooks = BB_CORNERS;
        m_queens = BB_D1 | BB_D8;
        m_kings = BB_E1 | BB_E8;

        m_promoted = BB_EMPTY;

        m_occupied_color[Color::WHITE] = BB_RANK_1 | BB_RANK_2;
        m_occupied_color[Color::BLACK] = BB_RANK_7 | BB_RANK_8;
        m_occupied = m_occupied_color[Color::WHITE] | m_occupied_color[Color::BLACK];
    }

    void clearBoard() {
        // Clears the board to a blank board with no pieces.

        m_pawns = m_knights = m_bishops = m_rooks = m_queens = m_kings = m_promoted = m_occupied =
            BB_EMPTY;
        m_occupied_color = {BB_EMPTY, BB_EMPTY};
    }

    Bitboard piecesMask(PieceType piece_type, Color color) const {
        Bitboard bb;
        switch (piece_type) {
        case PieceType::PAWN:
            bb = m_pawns;
            break;
        case PieceType::KNIGHT:
            bb = m_knights;
            break;
        case PieceType::BISHOP:
            bb = m_bishops;
            break;
        case PieceType::ROOK:
            bb = m_rooks;
            break;
        case PieceType::QUEEN:
            bb = m_queens;
            break;
        case PieceType::KING:
            bb = m_kings;
            break;
        case PieceType::NONE:
        case PieceType::NUM_PIECE_TYPES:
        default:
            // assert(false && "Unknown piece type");
            return BB_EMPTY;
        }
        return bb & m_occupied_color[color];
    }

    SquareSet pieces(PieceType piece_type, Color color) const {
        // Gets pieces of the given type and color.
        //
        // Returns a :class:`set of squares <chess::SquareSet>`.

        return SquareSet(piecesMask(piece_type, color));
    }

    std::optional<Piece> pieceAt(Square square) const {
        // Gets the :class:`piece <chess::Piece>` at the given square.

        auto piece_type = pieceTypeAt(square);
        if (piece_type.has_value()) {
            Bitboard mask = BB_SQUARES[square];
            Color color = (m_occupied_color[WHITE] & mask) ? WHITE : BLACK;
            return Piece(piece_type.value(), color);
        }
        return std::nullopt;
    }

    std::optional<PieceType> pieceTypeAt(Square square) const {
        // Gets the piece type at the given square.

        Bitboard mask = BB_SQUARES[square];

        if (!(m_occupied & mask)) {
            return std::nullopt;
        } else if (m_pawns & mask) {
            return PieceType::PAWN;
        } else if (m_knights & mask) {
            return PieceType::KNIGHT;
        } else if (m_bishops & mask) {
            return PieceType::BISHOP;
        } else if (m_rooks & mask) {
            return PieceType::ROOK;
        } else if (m_queens & mask) {
            return PieceType::QUEEN;
        } else {
            return PieceType::KING;
        }
    }

    std::optional<Color> colorAt(Square square) const {
        // Gets the color of the piece at the given square.

        Bitboard mask = BB_SQUARES[square];
        if (m_occupied_color[WHITE] & mask) {
            return WHITE;
        } else if (m_occupied_color[BLACK] & mask) {
            return BLACK;
        } else {
            return std::nullopt;
        }
    }

    std::optional<Square> king(Color color) const {
        // Finds the king square of the given side. Returns ``None`` if there
        // is no king of that color.
        //
        // In variants with king promotions, only non-promoted kings are
        // considered.

        Bitboard king_bb = m_kings & m_occupied_color[color] & ~m_promoted;
        if (king_bb) {
            return (Square) msb(king_bb);
        }
        return std::nullopt;
    }

    SquareSet attacks(Square square) const {
        // Gets the set of attackers of the given color for the given square.
        //
        // Pinned pieces still count as attackers.
        //
        // Returns a :class:`set of squares <chess::SquareSet>`.

        return SquareSet(_attacksMask(square));
    }

    Bitboard attackersMask(Color color, Square square) const {
        return _attackersMask(color, square, m_occupied);
    }

    bool isAttackedBy(Color color, Square square) const {
        // Checks if the given side attacks the given square.
        //
        // Pinned pieces still count as attackers. Pawns that can be captured
        // en passant are **not** considered attacked.
        return attackersMask(color, square) != 0;
    }

    SquareSet attackers(Color color, Square square) const {
        // Gets the set of attackers of the given color for the given square.
        //
        // Pinned pieces still count as attackers.
        //
        // Returns a :class:`set of squares <chess::SquareSet>`.

        return SquareSet(attackersMask(color, square));
    }

    SquareSet pin(Color color, Square square) const {
        // Detects an absolute pin (and its direction) of the given square to
        // the king of the given color.
        //
        // >>> import chess
        // >>>
        // >>> board = chess::Board("rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3
        // 7")
        // >>> board.isPinned(chess::WHITE, chess::C3)
        // True
        // >>> direction = board.pin(chess::WHITE, chess::C3)
        // >>> direction
        // SquareSet(0x0000_0001_0204_0810)
        // >>> print(direction)
        // . . . . . . . .
        // . . . . . . . .
        // . . . . . . . .
        // 1 . . . . . . .
        // . 1 . . . . . .
        // . . 1 . . . . .
        // . . . 1 . . . .
        // . . . . 1 . . .
        //
        // Returns a :class:`set of squares <chess::SquareSet>` that mask the rank,
        // file or diagonal of the pin. If there is no pin, then a mask of the
        // entire board is returned.
        return SquareSet(_pinMask(color, square));
    }

    bool isPinned(Color color, Square square) const {
        // Detects if the given square is pinned to the king of the given color.
        return _pinMask(color, square) != BB_ALL;
    }

    std::optional<Piece> removePieceAt(Square square) {
        auto color = (m_occupied_color[WHITE] & BB_SQUARES[square]) ? WHITE : BLACK;
        auto piece_type = _removePieceAt(square);
        return piece_type.has_value() ? std::optional<Piece>(Piece(piece_type.value(), color))
                                      : std::nullopt;
    }

    void setPieceAt(Square square, const std::optional<Piece> &piece, bool promoted = false) {
        if (!piece.has_value()) {
            removePieceAt(square);
        } else {
            _setPieceAt(square, piece->pieceType(), piece->color(), promoted);
        }
    }

    std::string boardFen() const {
        std::string board_fen = "";
        int empty = 0;

        for (int rank = 7; rank >= 0; --rank) {
            for (int file = 0; file < 8; ++file) {
                Square sq = square(file, rank);
                auto piece = pieceAt(sq);
                if (piece.has_value()) {
                    if (empty > 0) {
                        board_fen += std::to_string(empty);
                        empty = 0;
                    }
                    board_fen += piece->symbol();
                } else {
                    empty++;
                }
            }
            if (empty > 0) {
                board_fen += std::to_string(empty);
                empty = 0;
            }
            if (rank > 0) {
                board_fen += "/";
            }
        }

        return board_fen;
    }

    // Maps pieces by their squares using a mask to filter relevant squares.
    std::unordered_map<Square, Piece> pieceMap(Bitboard mask = BB_ALL) const {
        std::unordered_map<Square, Piece> result;
        for (Square square : SQUARES) {
            Bitboard squareBitboard = 1ULL << square;
            if (m_occupied & mask & squareBitboard) {
                auto piece = pieceAt(square);
                if (piece) {
                    result[square] = piece.value();
                }
            }
        }
        return result;
    }

    // Sets up the board from a map of pieces by square.
    void setPieceMap(const std::unordered_map<Square, Piece> &pieces) {
        clearBoard();
        for (const auto &[square, piece] : pieces) {
            setPieceAt(square, piece);
        }
    }

    // Returns a string representation of the board with Unicode pieces.
    std::string unicode(bool invert_color = false, bool borders = false,
                        const std::string &empty_square = "⭘", Color orientation = WHITE) const {
        std::stringstream builder;
        for (int rank = (orientation == WHITE ? 7 : 0);
             (orientation == WHITE ? rank >= 0 : rank < 8);
             (orientation == WHITE ? --rank : ++rank)) {
            if (borders) {
                builder << RANK_NAMES[rank] << " ";
            }

            for (int file = 0; file < 8; ++file) {
                Square sq = square(file, rank);

                auto piece = pieceAt(sq);
                if (piece.has_value()) {
                    builder << piece.value().unicode_symbol(invert_color);
                } else {
                    builder << empty_square;
                }

                if (borders)
                    builder << " ";
                else if (file < 7)
                    builder << " ";
            }

            if (borders || rank != (orientation == WHITE ? 0 : 7))
                builder << "\n";
        }

        if (borders) {
            builder << "  ";
            for (char c = 'a'; c <= 'h'; ++c) {
                builder << c << (c < 'h' ? " " : "");
            }
        }

        return builder.str();
    }

    // Returns the number of half-moves since the start of the game
    int ply() const { return 2 * (fullmove_number - 1) + (turn == BLACK ? 1 : 0); }

    bool isCheck() const { return _checkersMask() != BB_EMPTY; }

    bool isIntoCheck(const Move &move) {
        auto opt_king = king(turn);
        if (!opt_king.has_value())
            return false;

        Square king = opt_king.value();

        Bitboard checkers = attackersMask(!turn, king);
        if (checkers) {
            auto evasions = _generateEvasions(king, checkers, BB_SQUARES[move.fromSquare()],
                                              BB_SQUARES[move.toSquare()]);
            if (not contains(evasions, move)) {
                return true;
            }
        }

        return !isSafe(king, sliderBlockers(king), move);
    }

    bool isPseudoLegal(const Move &move) {
        // Check if a move is pseudo-legal
        if (!move)
            return false; // Null moves are not pseudo-legal

        auto piece = pieceTypeAt(move.fromSquare());
        if (!piece.has_value())
            return false; // Source square must not be vacant

        Bitboard from_mask = BB_SQUARES[move.fromSquare()];
        Bitboard to_mask = BB_SQUARES[move.toSquare()];

        if (!(m_occupied_color[turn] & from_mask))
            return false; // Check turn

        if (move.promotion()) {
            if (piece != PieceType::PAWN)
                return false;
            if (turn == WHITE && squareRank(move.toSquare()) != 7)
                return false;
            if (turn == BLACK && squareRank(move.toSquare()) != 0)
                return false;
        }

        if (piece == PieceType::KING) {
            auto castlingMoves = _generateCastlingMoves();
            if (contains(castlingMoves, move))
                return true; // Castling move is legal
        }

        if (m_occupied_color[turn] & to_mask)
            return false; // Destination square cannot be occupied by our piece

        if (piece == PieceType::PAWN) {
            // Specific pawn moves handling
            auto pawnMoves = _generatePseudoLegalMoves(from_mask, to_mask);
            return contains(pawnMoves, move);
        }

        return _attacksMask(move.fromSquare()) & to_mask;
    }

    bool isLegal(const Move &move) {
        // Check if a move is legal
        return isPseudoLegal(move) && !isIntoCheck(move);
    }

    bool isGameOver() {
        // Check if the game is over
        return outcome().has_value();
    }

    std::string result() {
        // Get the game result as a string
        auto outcomeOption = outcome();
        return outcomeOption ? outcomeOption->result() : "*";
    }

    std::optional<Outcome> outcome() {
        // Determine the outcome of the game
        if (isCheckmate())
            return Outcome(Termination::CHECKMATE, !turn);
        if (isInsufficientMaterial())
            return Outcome(Termination::INSUFFICIENT_MATERIAL, std::nullopt);
        if (_generateLegalMoves().empty())
            return Outcome(Termination::STALEMATE, std::nullopt);

        if (isSeventyFiveMoves())
            return Outcome(Termination::SEVENTYFIVE_MOVES, std::nullopt);

        return std::nullopt; // No outcome determined
    }

    bool isCheckmate() {
        if (!isCheck())
            return false;

        return _generateLegalMoves().empty();
    }

    bool isStalemate() {
        if (isCheck())
            return false;

        return _generateLegalMoves().empty();
    }

    bool isInsufficientMaterial() const {
        for (Color color : {WHITE, BLACK}) {
            if (!hasInsufficientMaterial(color))
                return false;
        }
        return true;
    }

    bool hasInsufficientMaterial(Color color) const {
        Bitboard our_pieces = m_occupied_color[color];
        if (our_pieces & (m_pawns | m_rooks | m_queens))
            return false;

        // Knights or bishops only
        if (our_pieces & m_knights) {
            return popcount(our_pieces) <= 2 && !(m_occupied_color[!color] & ~m_kings & ~m_queens);
        }

        if (our_pieces & m_bishops) {
            bool same_color_bishops =
                (!(m_bishops & BB_DARK_SQUARES)) || (!(m_bishops & BB_LIGHT_SQUARES));
            return same_color_bishops && !m_pawns && !m_knights;
        }

        return true;
    }

    bool isSeventyFiveMoves() { return _isHalfMoves(150); }

    bool isFiftyMoves() { return _isHalfMoves(100); }

    void push(Move move) {
        /*
        Updates the position with the given *move*.

        >>> Board board;
        >>>
        >>> Move Nf3 = Move::from_uci("g1f3");
        >>> board.push(Nf3)  # Make the move

        Null moves just increment the move counters, switch turns and forfeit
        en passant capturing.

        .. warning::
            Moves are not checked for legality. It is the caller's
            responsibility to ensure that the move is at least pseudo-legal or
            a null move.
        */

        // Updates the position with the given move
        // Reset the generated moves
        _resetStoredMoves();

        castling_rights = _cleanCastlingRights();

        // Reset en passant square
        auto epSquare = ep_square;
        ep_square = std::nullopt;

        // Increment move counters
        halfmove_clock++;
        if (turn == BLACK) {
            fullmove_number++;
        }

        // Handle null moves
        if (!move) {
            turn = !turn;
            return;
        }

        // Zero the half-move clock for pawn moves and captures
        if (isZeroing(move)) {
            halfmove_clock = 0;
        }

        Bitboard fromBB = BB_SQUARES[move.fromSquare()];
        Bitboard toBB = BB_SQUARES[move.toSquare()];

        bool promoted = (m_promoted & fromBB) != 0;
        auto piece = _removePieceAt(move.fromSquare());
        if (!piece.has_value()) {
            throw std::invalid_argument("Invalid move: " + move.uci() + " in position: " + fen());
        }
        auto pieceType = piece.value();

        Square captureSquare = move.toSquare();
        auto capturedPieceType = pieceTypeAt(captureSquare);

        // Update castling rights
        castling_rights &= ~fromBB & ~toBB;
        if (pieceType == PieceType::KING && !promoted) {
            if (turn == WHITE) {
                castling_rights &= ~BB_RANK_1;
            } else {
                castling_rights &= ~BB_RANK_8;
            }
        } else if (capturedPieceType == PieceType::KING && !(m_promoted & toBB)) {
            if (turn == WHITE && squareRank(captureSquare) == 7) {
                castling_rights &= ~BB_RANK_8;
            } else if (turn == BLACK && squareRank(captureSquare) == 0) {
                castling_rights &= ~BB_RANK_1;
            }
        }

        // Handle special pawn moves, including en passant captures
        if (pieceType == PieceType::PAWN) {
            int distance = move.toSquare() - move.fromSquare();

            if (distance == 16 && squareRank(move.fromSquare()) == 1) {
                ep_square = move.fromSquare() + 8;
            } else if (distance == -16 && squareRank(move.fromSquare()) == 6) {
                ep_square = move.fromSquare() - 8;
            } else if (epSquare.has_value() && move.toSquare() == epSquare.value() &&
                       (std::abs(distance) == 7 || std::abs(distance) == 9) &&
                       !capturedPieceType.has_value()) {
                // En passant capture
                int down = (turn == WHITE) ? -8 : 8;
                captureSquare = epSquare.value() + down;
                capturedPieceType = _removePieceAt(captureSquare);
            }
        }

        // Handle promotions
        if (move.promotion() != PieceType::NONE) {
            promoted = true;
            pieceType = move.promotion();
        }

        // Handle castling moves
        if (pieceType == PieceType::KING && (m_occupied_color[turn] & toBB)) {
            // Is a castling move
            bool aSide = squareFile(move.toSquare()) < squareFile(move.fromSquare());

            _removePieceAt(move.fromSquare());
            _removePieceAt(move.toSquare());

            if (aSide) {
                _setPieceAt((turn == WHITE) ? C1 : C8, PieceType::KING, turn);
                _setPieceAt((turn == WHITE) ? D1 : D8, PieceType::ROOK, turn);
            } else {
                _setPieceAt((turn == WHITE) ? G1 : G8, PieceType::KING, turn);
                _setPieceAt((turn == WHITE) ? F1 : F8, PieceType::ROOK, turn);
            }

        } else {
            // Is not a castling move
            _setPieceAt(move.toSquare(), pieceType, turn, promoted);
        }

        // Swap turns
        turn = !turn;
    }

    bool hasPseudoLegalEnPassant() {
        return ep_square.has_value() && !_generatePseudoLegalEP().empty();
    }

    bool hasLegalEnPassant() { return ep_square.has_value() && !_generateLegalEP().empty(); }

    bool isEnPassant(const Move &move) const {
        int distance = std::abs(move.toSquare() - move.fromSquare());
        return ep_square.has_value() && move.toSquare() == ep_square.value() &&
               (m_pawns & BB_SQUARES[move.fromSquare()]) && (distance == 7 || distance == 9) &&
               !(m_occupied & BB_SQUARES[move.toSquare()]);
    }

    bool isCapture(const Move &move) const {
        Bitboard touched = BB_SQUARES[move.fromSquare()] ^ BB_SQUARES[move.toSquare()];
        return (touched & m_occupied_color[!turn]) || isEnPassant(move);
    }

    bool isZeroing(const Move &move) const {
        Bitboard touched = BB_SQUARES[move.fromSquare()] ^ BB_SQUARES[move.toSquare()];
        return (touched & m_pawns) || (touched & m_occupied_color[!turn]);
    }

    bool reducesCastlingRights(const Move &move) const {
        Bitboard cr = _cleanCastlingRights();
        Bitboard touched = BB_SQUARES[move.fromSquare()] ^ BB_SQUARES[move.toSquare()];
        return (touched & cr) ||
               ((cr & BB_RANK_1) && touched & m_kings & m_occupied_color[WHITE] & ~m_promoted) ||
               ((cr & BB_RANK_8) && touched & m_kings & m_occupied_color[BLACK] & ~m_promoted);
    }

    bool isCastling(const Move &move) const {
        if (m_kings & BB_SQUARES[move.fromSquare()]) {
            int diff = squareFile(move.fromSquare()) - squareFile(move.toSquare());
            return std::abs(diff) > 1 ||
                   (m_rooks & m_occupied_color[turn] & BB_SQUARES[move.toSquare()]);
        }
        return false;
    }

    bool isKingsideCastling(const Move &move) const {
        return isCastling(move) && squareFile(move.toSquare()) > squareFile(move.fromSquare());
    }

    bool isQueensideCastling(const Move &move) const {
        return isCastling(move) && squareFile(move.toSquare()) < squareFile(move.fromSquare());
    }

    bool hasCastlingRights(Color color) const {
        Bitboard back_rank = (color == WHITE) ? BB_RANK_1 : BB_RANK_8;
        return bool(_cleanCastlingRights() & back_rank);
    }

    bool hasKingsideCastlingRights(Color color) const {
        Bitboard back_rank = (color == WHITE) ? BB_RANK_1 : BB_RANK_8;
        Bitboard kingMask = m_kings & m_occupied_color[color] & back_rank & ~m_promoted;
        if (kingMask == BB_EMPTY) {
            return false;
        }

        Bitboard castlingRights = _cleanCastlingRights() & back_rank;
        while (castlingRights) {
            // Isolate the least-significant bit (rightmost rook)
            Bitboard rook = lsb(castlingRights);

            // Kingside rook is to the right of the king
            if (rook > kingMask) {
                return true;
            }

            // Remove the least-significant bit
            castlingRights &= castlingRights - 1;
        }

        return false;
    }

    bool hasQueensideCastlingRights(Color color) const {
        Bitboard back_rank = (color == WHITE) ? BB_RANK_1 : BB_RANK_8;
        Bitboard kingMask = m_kings & m_occupied_color[color] & back_rank & ~m_promoted;
        if (kingMask == BB_EMPTY) {
            return false;
        }

        Bitboard castlingRights = _cleanCastlingRights() & back_rank;
        while (castlingRights) {
            // Isolate the least-significant bit (leftmost rook)
            Bitboard rook = lsb(castlingRights);

            // Queenside rook is to the left of the king
            if (rook < kingMask) {
                return true;
            }

            // Remove the least-significant bit
            castlingRights &= castlingRights - 1;
        }

        return false;
    }

    Status status() const {
        Status errors = Status::VALID;

        if (!m_occupied) {
            errors |= Status::EMPTY;
        }

        // Kings
        if (!(m_occupied_color[WHITE] & m_kings)) {
            errors |= Status::NO_WHITE_KING;
        }
        if (!(m_occupied_color[BLACK] & m_kings)) {
            errors |= Status::NO_BLACK_KING;
        }
        if (popcount(m_occupied & m_kings) > 2) {
            errors |= Status::TOO_MANY_KINGS;
        }

        // Piece counts
        if (popcount(m_occupied_color[WHITE]) > 16) {
            errors |= Status::TOO_MANY_WHITE_PIECES;
        }
        if (popcount(m_occupied_color[BLACK]) > 16) {
            errors |= Status::TOO_MANY_BLACK_PIECES;
        }

        // Pawn counts and positions
        if (popcount(m_occupied_color[WHITE] & m_pawns) > 8) {
            errors |= Status::TOO_MANY_WHITE_PAWNS;
        }
        if (popcount(m_occupied_color[BLACK] & m_pawns) > 8) {
            errors |= Status::TOO_MANY_BLACK_PAWNS;
        }
        if (m_pawns & BB_BACK_RANKS) {
            errors |= Status::PAWNS_ON_BACK_RANK;
        }

        // Castling rights
        if (castling_rights != _cleanCastlingRights()) {
            errors |= Status::BAD_CASTLING_RIGHTS;
        }

        // En passant
        auto valid_ep_square = validEPSquare();
        if (ep_square != valid_ep_square) {
            errors |= Status::INVALID_EP_SQUARE;
        }

        // Check conditions
        if (_wasIntoCheck()) {
            errors |= Status::OPPOSITE_CHECK;
        }

        Bitboard checkers = _checkersMask();
        if (popcount(checkers) > 2) {
            errors |= Status::TOO_MANY_CHECKERS;
        }

        // Impossible check
        Bitboard our_kings = m_kings & m_occupied_color[turn] & ~m_promoted;
        if (checkers && valid_ep_square != std::nullopt) {
            Bitboard pushed_to = valid_ep_square.value() ^ (turn == WHITE ? A2 : A7);
            Bitboard pushed_from = valid_ep_square.value() ^ (turn == WHITE ? A4 : A5);
            Bitboard occupied_before =
                (m_occupied & ~BB_SQUARES[pushed_to]) | BB_SQUARES[pushed_from];
            if (popcount(checkers) > 1 || ((Bitboard) msb(checkers) != pushed_to &&
                                           _attackedForKing(our_kings, occupied_before))) {
                errors |= Status::IMPOSSIBLE_CHECK;
            }
        } else {
            if (popcount(checkers) > 2 ||
                (popcount(checkers) == 2 && ray(lsb(checkers), msb(checkers)) & our_kings)) {
                errors |= Status::IMPOSSIBLE_CHECK;
            }
        }

        return errors;
    }

    std::optional<Square> validEPSquare() const {
        if (!ep_square.has_value())
            return std::nullopt;

        int ep_rank = (turn == WHITE) ? 5 : 2;
        Bitboard pawn_mask = (turn == WHITE) ? shiftDown(BB_SQUARES[ep_square.value()])
                                             : shiftUp(BB_SQUARES[ep_square.value()]);
        Bitboard seventh_rank_mask = (turn == WHITE) ? shiftUp(BB_SQUARES[ep_square.value()])
                                                     : shiftDown(BB_SQUARES[ep_square.value()]);

        if (squareRank(ep_square.value()) != ep_rank)
            return std::nullopt;
        if (!(m_pawns & m_occupied_color[!turn] & pawn_mask))
            return std::nullopt;
        if (m_occupied & BB_SQUARES[ep_square.value()])
            return std::nullopt;
        if (m_occupied & seventh_rank_mask)
            return std::nullopt;

        return ep_square;
    }

    bool isValid() const { return status() == Status::VALID; }

    bool epSkewered(Square king, Square capturer) const {
        if (!ep_square.has_value())
            return false;

        Square last_double = ep_square.value() + ((turn == WHITE) ? -8 : 8);
        Bitboard occupancy = m_occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] |
                             BB_SQUARES[ep_square.value()];

        Bitboard horizontal_attackers = m_occupied_color[!turn] & (m_rooks | m_queens);
        if (BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & occupancy] & horizontal_attackers) {
            return true;
        }

        Bitboard diagonal_attackers = m_occupied_color[!turn] & (m_bishops | m_queens);
        if (BB_DIAG_ATTACKS[king][BB_DIAG_MASKS[king] & occupancy] & diagonal_attackers) {
            return true;
        }

        return false;
    }

    Bitboard sliderBlockers(Square king) const {
        Bitboard rooks_and_queens = m_rooks | m_queens;
        Bitboard bishops_and_queens = m_bishops | m_queens;

        Bitboard snipers = ((BB_RANK_ATTACKS[king][0] & rooks_and_queens) |
                            (BB_FILE_ATTACKS[king][0] & rooks_and_queens) |
                            (BB_DIAG_ATTACKS[king][0] & bishops_and_queens)) &
                           m_occupied_color[!turn];

        Bitboard blockers = BB_EMPTY;
        for (Square sniper : scanReversed(snipers)) {
            Bitboard b = between(king, sniper) & m_occupied;
            if (b && BB_SQUARES[msb(b)] == b) {
                blockers |= b;
            }
        }

        return blockers & m_occupied_color[turn];
    }

    bool isSafe(Square king, Bitboard blockers, const Move &move) const {
        if (move.fromSquare() == king) {
            if (isCastling(move)) {
                return true;
            } else {
                return !isAttackedBy(!turn, move.toSquare());
            }
        } else if (isEnPassant(move)) {
            return (_pinMask(turn, move.fromSquare()) & BB_SQUARES[move.toSquare()]) &&
                   !epSkewered(king, move.fromSquare());
        } else {
            return !(blockers & BB_SQUARES[move.fromSquare()]) ||
                   (ray(move.fromSquare(), move.toSquare()) & BB_SQUARES[king]);
        }
    }

    bool operator==(const Board &other) const {
        return halfmove_clock == other.halfmove_clock && fullmove_number == other.fullmove_number &&
               m_pawns == other.m_pawns && m_knights == other.m_knights &&
               m_bishops == other.m_bishops && m_rooks == other.m_rooks &&
               m_queens == other.m_queens && m_kings == other.m_kings &&
               m_occupied == other.m_occupied && m_occupied_color == other.m_occupied_color &&
               m_promoted ==
                   other.m_promoted; //  && _transpositionKey() == other._transpositionKey();
    }

    Board copy() const {
        // Creates a copy of the board
        Board board(false);

        board.ep_square = ep_square;
        board.castling_rights = castling_rights;
        board.halfmove_clock = halfmove_clock;
        board.fullmove_number = fullmove_number;
        board.turn = turn;

        board.m_pawns = m_pawns;
        board.m_knights = m_knights;
        board.m_bishops = m_bishops;
        board.m_rooks = m_rooks;
        board.m_queens = m_queens;
        board.m_kings = m_kings;
        board.m_occupied = m_occupied;
        board.m_occupied_color = m_occupied_color;
        board.m_promoted = m_promoted;

        return board;
    }

    std::string fen() const {
        // Gets the FEN representation of the board
        std::stringstream builder;
        int empty = 0;

        for (int rank = 7; rank >= 0; --rank) {
            for (int file = 0; file < 8; ++file) {
                Square sq = square(file, rank);
                auto piece = pieceAt(sq);
                if (piece.has_value()) {
                    if (empty > 0) {
                        builder << empty;
                        empty = 0;
                    }
                    builder << piece->symbol();
                } else {
                    empty++;
                }
            }
            if (empty > 0) {
                builder << empty;
                empty = 0;
            }
            if (rank > 0) {
                builder << "/";
            }
        }

        builder << " " << (turn == WHITE ? "w" : "b") << " ";

        if (castling_rights == 0) {
            builder << "-";
        } else {
            if (castling_rights & BB_SQUARES[H1]) {
                builder << "K";
            }
            if (castling_rights & BB_SQUARES[A1]) {
                builder << "Q";
            }
            if (castling_rights & BB_SQUARES[H8]) {
                builder << "k";
            }
            if (castling_rights & BB_SQUARES[A8]) {
                builder << "q";
            }
        }

        builder << " " << (ep_square.has_value() ? squareName(ep_square.value()) : "-") << " ";
        builder << halfmove_clock << " " << fullmove_number;

        return builder.str();
    }

private:
    void _resetStoredMoves() {
        m_cachedLegalMoves = std::nullopt;
        m_cachedPseudoLegalMoves = std::nullopt;
        m_cachedLegalEPMoves = std::nullopt;
        m_cachedPseudoLegalEPMoves = std::nullopt;
    }

    Bitboard _attacksMask(Square square) const {
        Bitboard bb_square = BB_SQUARES[square];

        if (m_pawns & bb_square) {
            Color color = (m_occupied_color[WHITE] & bb_square) ? WHITE : BLACK;
            return BB_PAWN_ATTACKS[color][square];
        }
        if (m_knights & bb_square) {
            return BB_KNIGHT_ATTACKS[square];
        }
        if (m_kings & bb_square) {
            return BB_KING_ATTACKS[square];
        }

        Bitboard attacks = 0;
        if (m_bishops & bb_square || m_queens & bb_square) {
            attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & m_occupied];
        }
        if (m_rooks & bb_square || m_queens & bb_square) {
            attacks |= (BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & m_occupied] |
                        BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & m_occupied]);
        }
        return attacks;
    }

    Bitboard _attackersMask(Color color, Square square, Bitboard occupied) const {
        Bitboard rank_pieces = BB_RANK_MASKS[square] & occupied;
        Bitboard file_pieces = BB_FILE_MASKS[square] & occupied;
        Bitboard diag_pieces = BB_DIAG_MASKS[square] & occupied;

        Bitboard queens_and_rooks = m_queens | m_rooks;
        Bitboard queens_and_bishops = m_queens | m_bishops;

        Bitboard attackers =
            ((BB_KING_ATTACKS[square] & m_kings) | (BB_KNIGHT_ATTACKS[square] & m_knights) |
             (BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks) |
             (BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks) |
             (BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops) |
             (BB_PAWN_ATTACKS[!color][square] & m_pawns));

        return attackers & m_occupied_color[color];
    }

    Bitboard _pinMask(Color color, Square square) const {
        auto king = this->king(color);
        if (!king.has_value()) {
            return BB_ALL;
        }

        Bitboard square_mask = BB_SQUARES[square];

        std::array<std::pair<std::array<std::unordered_map<Bitboard, Bitboard>, 64> &, Bitboard>, 3>
            attacks_and_sliders = {{{BB_FILE_ATTACKS, m_rooks | m_queens},
                                    {BB_RANK_ATTACKS, m_rooks | m_queens},
                                    {BB_DIAG_ATTACKS, m_bishops | m_queens}}};

        for (auto &[attacks, sliders] : attacks_and_sliders) {
            Bitboard rays = attacks[king.value()][0];
            if (rays & square_mask) {
                Bitboard snipers = rays & sliders & m_occupied_color[!color];
                for (auto sniper : scanReversed(snipers)) {
                    if ((between(sniper, king.value()) & (m_occupied | square_mask)) ==
                        square_mask) {
                        return ray(king.value(), sniper);
                    }
                }
                break;
            }
        }

        return BB_ALL;
    }

    std::optional<PieceType> _removePieceAt(Square square) {
        auto piece_type = pieceTypeAt(square);
        Bitboard mask = BB_SQUARES[square];

        if (piece_type.has_value()) {
            switch (piece_type.value()) {
            case PieceType::PAWN:
                m_pawns ^= mask;
                break;
            case PieceType::KNIGHT:
                m_knights ^= mask;
                break;
            case PieceType::BISHOP:
                m_bishops ^= mask;
                break;
            case PieceType::ROOK:
                m_rooks ^= mask;
                break;
            case PieceType::QUEEN:
                m_queens ^= mask;
                break;
            case PieceType::KING:
                m_kings ^= mask;
                break;
            case PieceType::NONE:
            case PieceType::NUM_PIECE_TYPES:
            default:
                return std::nullopt;
            }

            m_occupied ^= mask;
            m_occupied_color[WHITE] &= ~mask;
            m_occupied_color[BLACK] &= ~mask;
            m_promoted &= ~mask;
        }

        return piece_type;
    }

    void _setPieceAt(Square square, PieceType piece_type, Color color, bool promoted = false) {
        removePieceAt(square); // Clear the square first

        Bitboard mask = BB_SQUARES[square];

        switch (piece_type) {
        case PieceType::PAWN:
            m_pawns |= mask;
            break;
        case PieceType::KNIGHT:
            m_knights |= mask;
            break;
        case PieceType::BISHOP:
            m_bishops |= mask;
            break;
        case PieceType::ROOK:
            m_rooks |= mask;
            break;
        case PieceType::QUEEN:
            m_queens |= mask;
            break;
        case PieceType::KING:
            m_kings |= mask;
            break;
        case PieceType::NONE:
        case PieceType::NUM_PIECE_TYPES:
        default:
            return;
        }

        m_occupied ^= mask;
        m_occupied_color[color] ^= mask;

        if (promoted) {
            m_promoted ^= mask;
        }
    }

    std::vector<Move> _generatePseudoLegalMoves(Bitboard from_mask = BB_ALL,
                                                Bitboard to_mask = BB_ALL) {
        if (m_cachedPseudoLegalMoves.has_value()) {
            return m_cachedPseudoLegalMoves.value();
        }

        std::vector<Move> moves;
        Bitboard our_pieces = m_occupied_color[turn];

        // Generate piece moves for non-pawn pieces
        Bitboard non_pawns = our_pieces & ~m_pawns & from_mask;

        for (Square from_square : scanReversed(non_pawns)) {
            Bitboard move_targets = _attacksMask(from_square) & ~our_pieces & to_mask;
            for (int to_square : scanReversed(move_targets)) {
                moves.push_back(Move(from_square, to_square));
            }
        }

        // Generate castling moves
        if (from_mask & m_kings) {
            auto castlingMoves = _generateCastlingMoves(from_mask, to_mask);
            extend(moves, castlingMoves);
        }

        // Generate pawn moves
        Bitboard pawns = m_pawns & m_occupied_color[turn] & from_mask;

        if (pawns == BB_EMPTY) {
            m_cachedPseudoLegalMoves = moves;
            return moves;
        }

        // Generate pawn captures
        for (int from_square : scanReversed(pawns)) {
            Bitboard targets =
                BB_PAWN_ATTACKS[turn][from_square] & m_occupied_color[!turn] & to_mask;
            for (Square to_square : scanReversed(targets)) {
                if (squareRank(to_square) == 0 || squareRank(to_square) == 7) { // Handle promotions
                    moves.push_back(Move(from_square, to_square, PieceType::QUEEN));
                    moves.push_back(Move(from_square, to_square, PieceType::ROOK));
                    moves.push_back(Move(from_square, to_square, PieceType::BISHOP));
                    moves.push_back(Move(from_square, to_square, PieceType::KNIGHT));
                } else {
                    moves.push_back(Move(from_square, to_square));
                }
            }
        }

        // Generate single and double pawn advances
        Bitboard single_moves, double_moves;
        if (turn == WHITE) {
            single_moves = shiftUp(pawns) & ~m_occupied;
            double_moves = shiftUp(single_moves) & ~m_occupied & (BB_RANK_3 | BB_RANK_4);
        } else {
            single_moves = shiftDown(pawns) & ~m_occupied;
            double_moves = shiftDown(single_moves) & ~m_occupied & (BB_RANK_6 | BB_RANK_5);
        }

        single_moves &= to_mask;
        double_moves &= to_mask;

        for (Square to_square : scanReversed(single_moves)) {
            Square from_square = to_square + (turn == BLACK ? 8 : -8);
            if (squareRank(to_square) == 0 || squareRank(to_square) == 7) { // Handle promotions
                moves.push_back(Move(from_square, to_square, PieceType::QUEEN));
                moves.push_back(Move(from_square, to_square, PieceType::ROOK));
                moves.push_back(Move(from_square, to_square, PieceType::BISHOP));
                moves.push_back(Move(from_square, to_square, PieceType::KNIGHT));
            } else {
                moves.push_back(Move(from_square, to_square));
            }
        }

        for (int to_square : scanReversed(double_moves)) {
            int from_square = to_square + (turn == BLACK ? 16 : -16);
            moves.push_back(Move(from_square, to_square));
        }

        // Generate en passant captures
        if (ep_square.has_value()) {
            auto epMoves = _generatePseudoLegalEP(from_mask, to_mask);
            extend(moves, epMoves);
        }

        m_cachedPseudoLegalMoves = moves;
        return moves;
    }

    std::vector<Move> _generatePseudoLegalEP(Bitboard from_mask = BB_ALL,
                                             Bitboard to_mask = BB_ALL) {
        if (m_cachedPseudoLegalEPMoves.has_value()) {
            return m_cachedPseudoLegalEPMoves.value();
        }

        if (!ep_square.has_value() || !(BB_SQUARES[ep_square.value()] & to_mask) ||
            (BB_SQUARES[ep_square.value()] & m_occupied)) {
            return {};
        }

        Bitboard capturers = m_pawns & m_occupied_color[turn] & from_mask &
                             BB_PAWN_ATTACKS[!turn][ep_square.value()] & BB_RANKS[turn ? 4 : 3];

        std::vector<Move> moves;
        moves.reserve(popcount(capturers));

        for (int capturer : scanReversed(capturers)) {
            moves.emplace_back(capturer, ep_square.value());
        }

        m_cachedPseudoLegalEPMoves = moves;
        return moves;
    }

    Bitboard _checkersMask() const {
        auto king = this->king(turn);
        if (!king.has_value())
            return BB_EMPTY;
        return attackersMask(!turn, king.value());
    }

    bool _wasIntoCheck() const {
        auto king = this->king(!turn);
        if (!king.has_value())
            return false;
        return isAttackedBy(turn, king.value());
    }

    Bitboard _cleanCastlingRights() const {
        Bitboard castling = castling_rights & m_rooks;
        Bitboard white_castling = castling & BB_RANK_1 & m_occupied_color[WHITE];
        Bitboard black_castling = castling & BB_RANK_8 & m_occupied_color[BLACK];

        white_castling &= BB_A1 | BB_H1;
        black_castling &= BB_A8 | BB_H8;

        if (!(m_occupied_color[WHITE] & m_kings & ~m_promoted & BB_E1))
            white_castling = BB_EMPTY;
        if (!(m_occupied_color[BLACK] & m_kings & ~m_promoted & BB_E8))
            black_castling = BB_EMPTY;

        return white_castling | black_castling;
    }

    bool _isHalfMoves(int n) { return halfmove_clock >= n && !_generateLegalMoves().empty(); }

    Move _findMove(int from_square, int to_square, PieceType promotion) {
        if (promotion == PieceType::NONE && (m_pawns & BB_SQUARES[from_square]) &&
            (BB_SQUARES[to_square] & BB_BACK_RANKS)) {
            promotion = PieceType::QUEEN;
        }

        Move move(from_square, to_square, promotion);
        if (!isLegal(move)) {
            throw std::runtime_error("No matching legal move found in the current position.");
        }

        return move;
    }

    std::vector<Move> _generateEvasions(Square king, Bitboard checkers, Bitboard from_mask = BB_ALL,
                                        Bitboard to_mask = BB_ALL) {
        std::vector<Move> evasions;
        Bitboard sliders = checkers & (m_bishops | m_rooks | m_queens);

        Bitboard attacked = 0;
        for (int checker : scanReversed(sliders)) {
            attacked |= ray(king, checker) & ~BB_SQUARES[checker];
        }

        if (BB_SQUARES[king] & from_mask) {
            for (int to_square : scanReversed(BB_KING_ATTACKS[king] & ~m_occupied_color[turn] &
                                              ~attacked & to_mask)) {
                evasions.emplace_back(king, to_square);
            }
        }

        int checker = msb(checkers);
        if (BB_SQUARES[checker] == checkers) {
            // Capture or block a single checker.
            Bitboard target = between(king, checker) | checkers;
            auto directEvasions = _generatePseudoLegalMoves(~m_kings & from_mask, target & to_mask);
            extend(evasions, directEvasions);

            // Capture the checking pawn en passant (avoiding duplicates).
            if (ep_square.has_value() && !(BB_SQUARES[ep_square.value()] & target)) {
                int last_double = ep_square.value() + (turn == WHITE ? -8 : 8);
                if (last_double == checker) {
                    auto epEvasions = _generatePseudoLegalEP(from_mask, to_mask);
                    extend(evasions, epEvasions);
                }
            }
        }

        return evasions;
    }

    std::vector<Move> _generateLegalMoves(Bitboard from_mask = BB_ALL, Bitboard to_mask = BB_ALL) {
        if (m_cachedLegalMoves.has_value()) {
            return m_cachedLegalMoves.value();
        }

        std::vector<Move> legalMoves;
        Bitboard king_mask = m_kings & m_occupied_color[turn];
        if (king_mask) {
            Square king = (Square) msb(king_mask);
            Bitboard blockers = sliderBlockers(king);
            Bitboard checkers = attackersMask(!turn, king);
            if (checkers) {
                auto evasions = _generateEvasions(king, checkers, from_mask, to_mask);
                legalMoves.reserve(evasions.size());
                for (const Move &move : evasions) {
                    if (isSafe(king, blockers, move)) {
                        legalMoves.push_back(move);
                    }
                }
            } else {
                auto pseudoLegalMoves = _generatePseudoLegalMoves(from_mask, to_mask);
                legalMoves.reserve(pseudoLegalMoves.size());
                for (const Move &move : pseudoLegalMoves) {
                    if (isSafe(king, blockers, move)) {
                        legalMoves.push_back(move);
                    }
                }
            }
        } else {
            legalMoves = _generatePseudoLegalMoves(from_mask, to_mask);
        }

        m_cachedLegalMoves = legalMoves;
        return legalMoves;
    }

    std::vector<Move> _generateLegalEP(Bitboard from_mask = BB_ALL, Bitboard to_mask = BB_ALL) {
        if (m_cachedLegalEPMoves.has_value()) {
            return m_cachedLegalEPMoves.value();
        }

        auto pseudoLegalEP = _generatePseudoLegalEP(from_mask, to_mask);

        std::vector<Move> legalEP;
        legalEP.reserve(pseudoLegalEP.size());
        for (const Move &move : pseudoLegalEP) {
            if (!isIntoCheck(move)) {
                legalEP.push_back(move);
            }
        }

        m_cachedLegalEPMoves = legalEP;
        return legalEP;
    }

    bool _attackedForKing(Bitboard path, Bitboard occupied) const {
        for (auto sq : scanReversed(path)) {
            if (_attackersMask(!turn, (Square) sq, occupied))
                return true;
        }
        return false;
    }

    std::vector<Move> _generateCastlingMoves(Bitboard from_mask = BB_ALL,
                                             Bitboard to_mask = BB_ALL) const {
        std::vector<Move> moves;

        Bitboard backrank = (turn == WHITE) ? BB_RANK_1 : BB_RANK_8;
        Bitboard king = m_occupied_color[turn] & m_kings & ~m_promoted & backrank & from_mask;
        king &= -king; // Isolate the least-significant bit (king)
        if (!king) {
            return moves;
        }

        Bitboard bb_c = BB_FILE_C & backrank;
        Bitboard bb_d = BB_FILE_D & backrank;
        Bitboard bb_f = BB_FILE_F & backrank;
        Bitboard bb_g = BB_FILE_G & backrank;

        for (auto candidate : scanReversed(_cleanCastlingRights() & backrank & to_mask)) {
            Bitboard rook = BB_SQUARES[candidate];

            bool a_side = rook < king;
            Bitboard king_to = a_side ? bb_c : bb_g;
            Bitboard rook_to = a_side ? bb_d : bb_f;

            Bitboard king_path = between(msb(king), msb(king_to));
            Bitboard rook_path = between(candidate, msb(rook_to));

            bool condition =
                (m_occupied ^ king ^ rook) & (king_path | rook_path | king_to | rook_to) ||
                _attackedForKing(king_path | king, m_occupied ^ king) ||
                _attackedForKing(king_to, m_occupied ^ king ^ rook ^ rook_to);

            if (!condition) {
                moves.emplace_back(candidate, msb(king_to));
            }
        }
        return moves;
    }

    using TranspositionKey = std::tuple<Bitboard, Bitboard, Bitboard, Bitboard, Bitboard, Bitboard,
                                        Bitboard, Bitboard, Color, Bitboard, std::optional<Square>>;

    TranspositionKey _transpositionKey() {
        return {m_pawns,
                m_knights,
                m_bishops,
                m_rooks,
                m_queens,
                m_kings,
                m_occupied_color[WHITE],
                m_occupied_color[BLACK],
                turn,
                _cleanCastlingRights(),
                hasLegalEnPassant() ? ep_square : std::optional<Square>{}};
    }
};
} // namespace chess

/* The following code is a small test suite for the bitboard functions.

#include <array>
#include <cstdint>
#include <iostream>

void testLsbMsb() {
    struct Test {
        uint64_t input;
        int expectedLsb;
        int expectedMsb;
    };

    std::array<Test, 8> tests = {{
        {0b1, 0, 0},
        {0b10, 1, 1},
        {0b100, 2, 2},
        {0b1000, 3, 3},
        {0b10000, 4, 4},
        {0xFFFFFFFFFFFFFFFF, 0, 63},
        {0x8000000000000000, 63, 63},
        {0, -1, -1} // Edge case: no bits set
    }};

    bool passed = true;
    for (const auto &test : tests) {
        int lsbResult = lsb(test.input);
        int msbResult = msb(test.input);
        if (lsbResult != test.expectedLsb || msbResult != test.expectedMsb) {
            std::cerr << "Test failed for input: " << test.input << "\n"
                      << "  Expected LSB: " << test.expectedLsb << ", got: " << lsbResult << "\n"
                      << "  Expected MSB: " << test.expectedMsb << ", got: " << msbResult << "\n";
            passed = false;
        }
    }

    if (passed) {
        std::cerr << "All tests passed!\n";
    } else {
        std::cerr << "Some tests failed.\n";
    }
}

void testPopcount() {
    struct Test {
        uint64_t input;
        int expected;
    };

    std::vector<Test> tests = {
        {0x0, 0},                 // No bits set
        {0x1, 1},                 // LSB set
        {0x8000000000000000, 1},  // MSB set
        {0xFFFFFFFFFFFFFFFF, 64}, // All bits set
        {0xAAAAAAAAAAAAAAAA, 32}, // Alternate bits set
        {0x5555555555555555, 32}, // Alternate bits set, starting with LSB
        {0x0F0F0F0F0F0F0F0F, 32}, // Nibble patterns
        {0x3333333333333333, 32}, // Pair patterns
        {0x0101010101010101, 8},  // Every 8th bit set
        {0x1249249249249249, 21}, // Sparse bits set
    };

    bool allTestsPassed = true;

    for (const auto &test : tests) {
        int result = popcount(test.input);
        if (result != test.expected) {
            std::cerr << "Test failed for input: " << test.input << "\n"
                      << "  Expected: " << test.expected << ", got: " << result << "\n";
            allTestsPassed = false;
        }
    }

    if (allTestsPassed) {
        std::cerr << "All tests passed!\n";
    } else {
        std::cerr << "Some tests failed.\n";
    }
}

int main() {
    testLsbMsb();
    testPopcount();
    return 0;
}

*/

// perft tests for chess
int perft(chess::Board &board, int depth) {
    if (depth == 0) {
        return 1;
    }

    int count = 0;
    auto moves = board.legalMoves();
    for (const auto &move : moves) {
        chess::Board newBoard = board.copy();
        newBoard.push(move);
        count += perft(newBoard, depth - 1);
    }
    return count;
}

int main() {
    chess::Board board =
        chess::Board::fromFEN("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1");

    for (int depth = 1; depth <= 5; ++depth) {
        int result = perft(board, depth);
        std::cout << "Perft result at depth " << depth << ": " << result << std::endl;
    }

    /*
    Perft result at depth 1: 20
    Perft result at depth 2: 400
    Perft result at depth 3: 8888
    Perft result at depth 4: 196264
    Perft result at depth 5: 4812803


    WARNING: This is wrong an indicates an error in the move generation code. At depth 5, the
    number of moves should be: 4865609!
    TODO: Fix chess implementation to pass the perft test.
    */

    return 0;
}