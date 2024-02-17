#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace chess {
enum Color : bool { WHITE, BLACK };
constexpr Color operator!(Color color) {
    return color == Color::WHITE ? Color::BLACK : Color::WHITE;
}

constexpr std::array<Color, 2> COLORS = {Color::WHITE, Color::BLACK};
const std::array<std::string, 2> COLOR_NAMES = {"black", "white"};

enum class PieceType { NONE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
constexpr std::array<PieceType, 7> PIECE_TYPES = {PieceType::PAWN,   PieceType::KNIGHT,
                                                  PieceType::BISHOP, PieceType::ROOK,
                                                  PieceType::QUEEN,  PieceType::KING};
const std::array<std::string, 7> PIECE_SYMBOLS = {" ", "p", "n", "b", "r", "q", "k"};
const std::array<std::string, 7> PIECE_NAMES = {"none", "pawn",  "knight", "bishop",
                                                "rook", "queen", "king"};
std::string pieceSymbol(PieceType piece_type) {
    return PIECE_SYMBOLS[static_cast<int>(piece_type)];
}

std::string pieceName(PieceType piece_type) { return PIECE_NAMES[static_cast<int>(piece_type)]; }
const std::unordered_map<char, std::string> UNICODE_PIECE_SYMBOLS = {
    {'R', "♖"}, {'r', "♜"}, {'N', "♘"}, {'n', "♞"}, {'B', "♗"}, {'b', "♝"},
    {'Q', "♕"}, {'q', "♛"}, {'K', "♔"}, {'k', "♚"}, {'P', "♙"}, {'p', "♟"},
};
const std::array<std::string, 8> FILE_NAMES = {"a", "b", "c", "d", "e", "f", "g", "h"};
const std::array<std::string, 8> RANK_NAMES = {"1", "2", "3", "4", "5", "6", "7", "8"};

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

constexpr Status operator|(Status a, Status b) {
    return (Status) (static_cast<int>(a) | static_cast<int>(b));
}
constexpr Status operator|=(Status &a, Status b) {
    a = a | b;
    return a;
}
constexpr Status operator&(Status a, Status b) {
    return (Status) (static_cast<int>(a) & static_cast<int>(b));
}
constexpr Status operator&=(Status &a, Status b) {
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

    std::string result() const {
        if (!winner.has_value()) {
            return "1/2-1/2";
        }
        return winner == Color::WHITE ? "1-0" : "0-1";
    }
};

class InvalidMoveError : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

class IllegalMoveError : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
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

constexpr int operator+(Square square) { return static_cast<int>(square); }
constexpr int operator-(Square square) { return -static_cast<int>(square); }
constexpr Square operator+(Square square, int n) { return static_cast<Square>(static_cast<int>(square) + n); }
constexpr Square operator-(Square square, int n) { return static_cast<Square>(static_cast<int>(square) - n); }
constexpr Square &operator+=(Square &square, int n) { return square = square + n; }
constexpr Square &operator-=(Square &square, int n) { return square = square - n; }

constexpr std::array<Square, 64> SQUARES = {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
};

const std::array<std::string, 64> SQUARE_NAMES = {
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

int parseSquare(const std::string &name) {
    auto it = std::find(SQUARE_NAMES.begin(), SQUARE_NAMES.end(), name);
    if (it == SQUARE_NAMES.end()) {
        throw std::invalid_argument("Invalid square name");
    }
    return std::distance(SQUARE_NAMES.begin(), it);
}

std::string squareName(Square square) {
    if (square < 0 || square >= 64) {
        throw std::invalid_argument("Square index out of bounds");
    }
    return SQUARE_NAMES[square];
}

Square square(int file_index, int rank_index) { return (Square) (rank_index * 8 + file_index); }

Square squareFile(Square square) { return (Square) (square & 7); }

Square squareRank(Square square) { return (Square) (square >> 3); }

constexpr Square squareMirror(Square square) { return (Square) (square ^ 0x38); }

constexpr std::array<Square, 64> SQUARES_180 = [] {
    std::array<Square, 64> squares_180;
    for (size_t i = 0; i < SQUARES.size(); ++i) {
        squares_180[i] = squareMirror(SQUARES[i]);
    }
    return squares_180;
}();

using Bitboard = unsigned long long;
constexpr Bitboard BB_EMPTY = 0;
constexpr Bitboard BB_ALL = 0xFFFF'FFFF'FFFF'FFFF;

constexpr std::array<Bitboard, 64> BB_SQUARES = [] {
    std::array<Bitboard, 64> squares;
    for (int i = 0; i < 64; ++i) {
        squares[i] = 1ULL << i;
    }
    return squares;
}();

// clang-format off
enum BB_SQUARE {
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

constexpr Bitboard BB_CORNERS = BB_A1 | BB_H1 | BB_A8 | BB_H8;
constexpr Bitboard BB_CENTER = BB_D4 | BB_E4 | BB_D5 | BB_E5;

constexpr Bitboard BB_LIGHT_SQUARES = 0x55AA'55AA'55AA'55AA;
constexpr Bitboard BB_DARK_SQUARES = 0xAA55'AA55'AA55'AA55;

constexpr std::array<Bitboard, 8> BB_FILES = [] {
    std::array<Bitboard, 8> files;
    for (int i = 0; i < 8; ++i) {
        files[i] = 0x0101'0101'0101'0101ULL << i;
    }
    return files;
}();

// clang-format off
enum BB_FILE {
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

constexpr std::array<Bitboard, 8> BB_RANKS = [] {
    std::array<Bitboard, 8> ranks;
    for (int i = 0; i < 8; ++i) {
        ranks[i] = 0xFFULL << (8 * i);
    }
    return ranks;
}();

// clang-format off
enum BB_RANK {
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

constexpr Bitboard BB_BACK_RANKS = BB_RANK_1 | BB_RANK_8;

int squareDistance(Square a, Square b) {
    return std::max(std::abs(squareFile(a) - squareFile(b)),
                    std::abs(squareRank(a) - squareRank(b)));
}

int squareManhattanDistance(Square a, Square b) {
    return std::abs(squareFile(a) - squareFile(b)) + std::abs(squareRank(a) - squareRank(b));
}

int squareKnightDistance(Square a, Square b) {
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

    int m = std::ceil(std::max({dx / 2.0, dy / 2.0, (dx + dy) / 3.0}));
    return m + ((m + dx + dy) % 2);
}

int lsb(Bitboard bb) {
    if (bb == 0)
        return -1;
    return std::countr_zero(bb);
}

std::vector<int> scanForward(Bitboard bb) {
    std::vector<int> squares;
    while (bb) {
        int sq = lsb(bb);
        squares.push_back(sq);
        bb &= bb - 1; // Reset LSB
    }
    return squares;
}

int msb(Bitboard bb) {
    if (bb == 0)
        return -1;
    return 63 - std::countl_zero(bb);
}

std::vector<Square> scanReversed(Bitboard bb) {
    std::vector<Square> squares;
    while (bb) {
        int sq = msb(bb);
        squares.push_back((Square) sq);
        bb &= ~(1ULL << sq); // Reset MSB
    }
    return squares;
}

int popcount(Bitboard bb) { return std::popcount(bb); }

Bitboard flipVertical(Bitboard bb) {
    bb = ((bb >> 8) & 0x00FF'00FF'00FF'00FF) | ((bb & 0x00FF'00FF'00FF'00FF) << 8);
    bb = ((bb >> 16) & 0x0000'FFFF'0000'FFFF) | ((bb & 0x0000'FFFF'0000'FFFF) << 16);
    bb = (bb >> 32) | ((bb & 0x0000'0000'FFFF'FFFF) << 32);
    return bb;
}

Bitboard flipHorizontal(Bitboard bb) {
    bb = ((bb >> 1) & 0x5555'5555'5555'5555) | ((bb & 0x5555'5555'5555'5555) << 1);
    bb = ((bb >> 2) & 0x3333'3333'3333'3333) | ((bb & 0x3333'3333'3333'3333) << 2);
    bb = ((bb >> 4) & 0x0F0F'0F0F'0F0F'0F0F) | ((bb & 0x0F0F'0F0F'0F0F'0F0F) << 4);
    return bb;
}

Bitboard flipDiagonal(Bitboard bb) {
    Bitboard t;
    t = (bb ^ (bb << 28)) & 0x0F0F'0F0F'0000'0000ULL;
    bb = bb ^ t ^ (t >> 28);
    t = (bb ^ (bb << 14)) & 0x3333'0000'3333'0000ULL;
    bb = bb ^ t ^ (t >> 14);
    t = (bb ^ (bb << 7)) & 0x5500'5500'5500'5500ULL;
    bb = bb ^ t ^ (t >> 7);
    return bb;
}

Bitboard flipAntiDiagonal(Bitboard bb) {
    Bitboard t;
    t = bb ^ (bb << 36);
    bb = bb ^ ((t ^ (bb >> 36)) & 0xF0F0'F0F0'0F0F'0F0FULL);
    t = (bb ^ (bb << 18)) & 0xCCCC'0000'CCCC'0000ULL;
    bb = bb ^ t ^ (t >> 18);
    t = (bb ^ (bb << 9)) & 0xAA00'AA00'AA00'AA00ULL;
    bb = bb ^ t ^ (t >> 9);
    return bb;
}

Bitboard shiftDown(Bitboard b) { return b >> 8; }

Bitboard shiftUp(Bitboard b) { return (b << 8) & BB_ALL; }

Bitboard _slidingAttacks(Square square, Bitboard occupied, const std::vector<int> &deltas) {
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

std::array<Bitboard, 64> BB_KNIGHT_ATTACKS = [] {
    std::array<Bitboard, 64> attacks;
    for (Square sq : SQUARES) {
        attacks[sq] = _slidingAttacks(sq, BB_ALL, {17, 15, 10, 6, -17, -15, -10, -6});
    }
    return attacks;
}();

std::array<Bitboard, 64> BB_KING_ATTACKS = [] {
    std::array<Bitboard, 64> attacks;
    for (Square sq : SQUARES) {
        attacks[sq] = _slidingAttacks(sq, BB_ALL, {9, 8, 7, 1, -9, -8, -7, -1});
    }
    return attacks;
}();

std::array<std::array<Bitboard, 64>, 2> BB_PAWN_ATTACKS = [] {
    std::array<std::array<Bitboard, 64>, 2> attacks;
    for (int color = 0; color < 2; ++color) {
        for (Square sq : SQUARES) {
            attacks[color][sq] = _slidingAttacks(
                sq, BB_ALL, color == 0 ? std::vector<int>{-7, -9} : std::vector<int>{7, 9});
        }
    }
    return attacks;
}();

Bitboard _edges(Square square) {
    return (((BB_RANKS[0] | BB_RANKS[7]) & ~BB_RANKS[squareRank(square)]) |
            ((BB_FILES[0] | BB_FILES[7]) & ~BB_FILES[squareFile(square)]));
}

std::vector<Bitboard> _carryRippler(Bitboard mask) {
    std::vector<Bitboard> subsets;
    Bitboard subset = BB_EMPTY;
    do {
        subsets.push_back(subset);
        subset = (subset - mask) & mask;
    } while (subset != BB_EMPTY);
    return subsets;
}

std::pair<std::array<Bitboard, 64>, std::array<std::unordered_map<Bitboard, Bitboard>, 64>>
_attack_table(const std::vector<int> &deltas) {
    std::array<Bitboard, 64> mask_table;
    std::array<std::unordered_map<Bitboard, Bitboard>, 64> attack_table;

    for (Square square : SQUARES) {
        std::unordered_map<Bitboard, Bitboard> attacks;

        Bitboard mask = _slidingAttacks(square, 0, deltas) & ~_edges(square);
        for (auto subset : _carryRippler(mask)) {
            attacks[subset] = _slidingAttacks(square, subset, deltas);
        }

        attack_table[square] = attacks;
        mask_table[square] = mask;
    }

    return {mask_table, attack_table};
}

auto [BB_DIAG_MASKS, BB_DIAG_ATTACKS] = _attack_table({-9, -7, 7, 9});
auto [BB_FILE_MASKS, BB_FILE_ATTACKS] = _attack_table({-8, 8});
auto [BB_RANK_MASKS, BB_RANK_ATTACKS] = _attack_table({-1, 1});

std::array<std::array<Bitboard, 64>, 64> BB_RAYS = [] {
    std::array<std::array<Bitboard, 64>, 64> rays;
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

Bitboard ray(int a, int b) { return BB_RAYS[a][b]; }

Bitboard between(int a, int b) {
    Bitboard bb = BB_RAYS[a][b] & ((BB_ALL << a) ^ (BB_ALL << b));
    return bb & (bb - 1);
}

class Piece {
public:
    // Constructors
    Piece() : value(0) {}

    Piece(PieceType piece_type, Color color)
        : value((static_cast<unsigned char>(piece_type) << 1) | static_cast<unsigned char>(color)) {
    }

    Piece operator=(const Piece &other) { return Piece(other.pieceType(), other.color()); }

    // Get the piece type
    PieceType pieceType() const { return static_cast<PieceType>(value >> 1); }

    // Get the piece color
    Color color() const { return static_cast<Color>(value & 1); }

    // Get the symbol of the piece
    std::string symbol() const {
        auto symbol = pieceSymbol(pieceType());
        return color() == Color::WHITE ? symbol : std::string(1, tolower(symbol[0]));
    }

    // Get the Unicode symbol of the piece
    std::string unicode_symbol(bool invert_color = false) const {
        char symbolKey = symbol()[0];
        if (invert_color) {
            symbolKey = isupper(symbolKey) ? tolower(symbolKey) : toupper(symbolKey);
        }
        return UNICODE_PIECE_SYMBOLS.at(symbolKey);
    }

    // Overload hash function
    size_t hash() const { return value; }

    // Class method to create Piece from symbol
    static Piece from_symbol(char symbol) {
        auto it = std::find_if(
            PIECE_SYMBOLS.begin(), PIECE_SYMBOLS.end(),
            [symbol](const std::string &s) { return tolower(s[0]) == tolower(symbol); });
        if (it != PIECE_SYMBOLS.end()) {
            int index = std::distance(PIECE_SYMBOLS.begin(), it);
            return Piece(static_cast<PieceType>(index),
                         isupper(symbol) ? Color::WHITE : Color::BLACK);
        }
        throw std::invalid_argument("Invalid symbol for piece creation");
    }

private:
    // Encoded as: 0bPPPC where PPP = PieceType (3 bits), C = Color (1 bit)
    const unsigned char value;
};

class Move {
public:
    Move() : value(0) {}

    Move(int from_square, int to_square, std::optional<PieceType> promotion = std::nullopt)
        : value(getValue(from_square, to_square, promotion)) {}

    Move operator=(const Move &other) {
        return Move(other.fromSquare(), other.toSquare(), other.promotion());
    }

    Square fromSquare() const { return (Square) ((value & FROM_SQUARE_MASK) >> FROM_SQUARE_SHIFT); }

    Square toSquare() const { return (Square) ((value & TO_SQUARE_MASK) >> TO_SQUARE_SHIFT); }

    std::optional<PieceType> promotion() const {
        PieceType type = static_cast<PieceType>((value & PROMOTION_MASK) >> PROMOTION_SHIFT);
        if (type != PieceType::NONE) {
            return type;
        }
        return std::nullopt;
    }

    std::string uci() const {
        if (!*this)
            return "0000";

        std::string uci = SQUARE_NAMES[fromSquare()] + SQUARE_NAMES[toSquare()];
        auto promo = promotion();
        if (promo) {
            uci += pieceSymbol(promo.value());
        }
        return uci;
    }

    bool operator!() const { return value == 0; }

    bool operator==(const Move &other) const { return value == other.value; }

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
        std::optional<PieceType> promotion = std::nullopt;
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

    int getValue(int from_square, int to_square,
                 std::optional<PieceType> promotion = std::nullopt) {
        int value = 0;
        value |= (from_square << FROM_SQUARE_SHIFT);
        value |= (to_square << TO_SQUARE_SHIFT);
        if (promotion) {
            value |= (static_cast<unsigned short>(promotion.value()) << PROMOTION_SHIFT);
        }
        return value;
    }

    const unsigned short value;
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
    std::size_t size() const { return popcount(mask); }

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
    std::vector<Square> squares() const {
        std::vector<Square> result;
        for (Square square : SQUARES) {
            if (contains(square)) {
                result.push_back(square);
            }
        }
        return result;
    }

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
    std::array<Bitboard, 2> occupied_co; // Bitboards for white and black pieces
    Bitboard pawns, knights, bishops, rooks, queens, kings, promoted, occupied;

    std::optional<std::vector<Move>> cachedLegalMoves = std::nullopt;
    std::optional<std::vector<Move>> cachedPseudoLegalMoves = std::nullopt;
    std::optional<std::vector<Move>> cachedLegalEPMoves = std::nullopt;
    std::optional<std::vector<Move>> cachedPseudoLegalEPMoves = std::nullopt;

public:
    Board(bool starting_position = true)
        : ep_square(std::nullopt), fullmove_number(1), halfmove_clock(0), turn(WHITE),
          castling_rights(BB_CORNERS), promoted(BB_EMPTY) {
        if (starting_position) {
            resetBoard();
        } else {
            clearBoard();
        }
    }

    std::vector<Move> legalMoves() {
        // A dynamic list of legal moves.
        return _generateLegalMoves();
    }

    std::vector<Move> pseudoLegalMoves() {
        // A dynamic list of pseudo-legal moves, much like the legal move list.
        //
        // Pseudo-legal moves might leave or put the king in check, but are
        // otherwise valid. Null moves are not pseudo-legal. Castling moves are
        // only included if they are completely legal.
        return _generatePseudoLegalMoves();
    }

    void resetBoard() {
        // Resets pieces to the starting position.

        pawns = BB_RANK_2 | BB_RANK_7;
        knights = BB_B1 | BB_G1 | BB_B8 | BB_G8;
        bishops = BB_C1 | BB_F1 | BB_C8 | BB_F8;
        rooks = BB_CORNERS;
        queens = BB_D1 | BB_D8;
        kings = BB_E1 | BB_E8;

        promoted = BB_EMPTY;

        occupied_co[Color::WHITE] = BB_RANK_1 | BB_RANK_2;
        occupied_co[Color::BLACK] = BB_RANK_7 | BB_RANK_8;
        occupied = occupied_co[Color::WHITE] | occupied_co[Color::BLACK];
    }

    void clearBoard() {
        // Clears the board to a blank board with no pieces.

        pawns = knights = bishops = rooks = queens = kings = promoted = occupied = BB_EMPTY;
        occupied_co = {BB_EMPTY, BB_EMPTY};
    }

    Bitboard piecesMask(PieceType piece_type, Color color) const {
        Bitboard bb;
        switch (piece_type) {
        case PieceType::PAWN:
            bb = pawns;
            break;
        case PieceType::KNIGHT:
            bb = knights;
            break;
        case PieceType::BISHOP:
            bb = bishops;
            break;
        case PieceType::ROOK:
            bb = rooks;
            break;
        case PieceType::QUEEN:
            bb = queens;
            break;
        case PieceType::KING:
            bb = kings;
            break;
        default:
            assert(false && "Unknown piece type");
            return BB_EMPTY;
        }
        return bb & occupied_co[color];
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
            Color color = (occupied_co[WHITE] & mask) ? WHITE : BLACK;
            return Piece(piece_type.value(), color);
        }
        return std::nullopt;
    }

    std::optional<PieceType> pieceTypeAt(Square square) const {
        // Gets the piece type at the given square.

        Bitboard mask = BB_SQUARES[square];

        if (!(occupied & mask)) {
            return std::nullopt;
        } else if (pawns & mask) {
            return PieceType::PAWN;
        } else if (knights & mask) {
            return PieceType::KNIGHT;
        } else if (bishops & mask) {
            return PieceType::BISHOP;
        } else if (rooks & mask) {
            return PieceType::ROOK;
        } else if (queens & mask) {
            return PieceType::QUEEN;
        } else {
            return PieceType::KING;
        }
    }

    std::optional<Color> colorAt(Square square) const {
        // Gets the color of the piece at the given square.

        Bitboard mask = BB_SQUARES[square];
        if (occupied_co[WHITE] & mask) {
            return WHITE;
        } else if (occupied_co[BLACK] & mask) {
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

        Bitboard king_bb = kings & occupied_co[color] & ~promoted;
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

    bool isAttackedBy(Color color, Square square) const {
        // Checks if the given side attacks the given square.
        //
        // Pinned pieces still count as attackers. Pawns that can be captured
        // en passant are **not** considered attacked.
        return _attackersMask(color, square) != 0;
    }

    SquareSet attackers(Color color, Square square) const {
        // Gets the set of attackers of the given color for the given square.
        //
        // Pinned pieces still count as attackers.
        //
        // Returns a :class:`set of squares <chess::SquareSet>`.

        return SquareSet(_attackersMask(color, square));
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
        auto color = (occupied_co[WHITE] & BB_SQUARES[square]) ? WHITE : BLACK;
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
    std::unordered_map<Square, Piece> pieceMap(Bitboard mask = BB_ALL) {
        std::unordered_map<Square, Piece> result;
        for (Square square : SQUARES) {
            Bitboard squareBitboard = 1ULL << square;
            if (occupied & mask & squareBitboard) {
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
                        const std::string &empty_square = "⭘", Color orientation = WHITE) {
        std::stringstream builder;
        for (int rank = (orientation == WHITE ? 7 : 0);
             (orientation == WHITE ? rank >= 0 : rank < 8);
             (orientation == WHITE ? --rank : ++rank)) {
            if (borders) {
                builder << "  " << std::string(17, '-') << "\n" << RANK_NAMES[rank] << " ";
            }

            for (int file = 0; file < 8; ++file) {
                Square sq = square(file, rank);
                if (borders && file == 0)
                    builder << "|";

                auto piece = pieceAt(sq);
                if (piece.has_value()) {
                    builder << (invert_color ? piece.value().unicode_symbol(true)
                                             : piece.value().unicode_symbol());
                } else {
                    builder << empty_square;
                }

                if (borders)
                    builder << "|";
                else if (file < 7)
                    builder << " ";
            }

            if (borders || rank != (orientation == WHITE ? 0 : 7))
                builder << "\n";
        }

        if (borders) {
            builder << "  " << std::string(17, '-') << "\n   ";
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
        auto opt_king = this->king(turn);
        if (!opt_king.has_value())
            return false;

        Square king = opt_king.value();

        Bitboard checkers = _attackersMask(!turn, king);
        if (checkers) {
            auto evasions = _generateEvasions(king, checkers, BB_SQUARES[move.fromSquare()],
                                              BB_SQUARES[move.toSquare()]);
            if (std::find(evasions.begin(), evasions.end(), move) == evasions.end()) {
                return true;
            }
        }

        return !isSafe(king, sliderBlockers(king), move);
    }

    bool isPseudoLegal(const Move &move) {
        // Check if a move is pseudo-legal
        if (!move)
            return false; // Null moves are not pseudo-legal in C++ context

        auto piece = pieceTypeAt(move.fromSquare());
        if (!piece.has_value())
            return false; // Source square must not be vacant

        Bitboard from_mask = BB_SQUARES[move.fromSquare()];
        Bitboard to_mask = BB_SQUARES[move.toSquare()];

        if (!(occupied_co[turn] & from_mask))
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
            auto castlingMoves = _generateCastlingMoves(); // This needs to be defined
            return std::find(castlingMoves.begin(), castlingMoves.end(), move) !=
                   castlingMoves.end();
        }

        if (occupied_co[turn] & to_mask)
            return false; // Destination square cannot be occupied by our piece

        if (piece == PieceType::PAWN) {
            // Specific pawn moves handling
            auto pawnMoves = _generatePseudoLegalMoves(from_mask, to_mask);
            return std::find(pawnMoves.begin(), pawnMoves.end(), move) != pawnMoves.end();
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
        Bitboard our_pieces = occupied_co[color];
        if (our_pieces & (pawns | rooks | queens))
            return false;

        // Knights or bishops only
        if (our_pieces & knights) {
            return popcount(our_pieces) <= 2 && !(occupied_co[!color] & ~(kings | queens));
        }

        if (our_pieces & bishops) {
            bool same_color_bishops =
                !((bishops & BB_DARK_SQUARES) && (bishops & BB_LIGHT_SQUARES));
            return same_color_bishops && !pawns && !knights;
        }

        return true;
    }

    bool isSeventyFiveMoves() { return _isHalfMoves(150); }

    bool isFiftyMoves() { return _isHalfMoves(100); }

    void resetStoredMoves() {
        cachedLegalMoves = std::nullopt;
        cachedPseudoLegalMoves = std::nullopt;
        cachedLegalEPMoves = std::nullopt;
        cachedPseudoLegalEPMoves = std::nullopt;
    }

    void push(const Move &move) {
        // Updates the position with the given move
        // Reset the generated moves
        resetStoredMoves();

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
        if (pieceTypeAt(move.fromSquare()) == PieceType::PAWN || isCapture(move)) {
            halfmove_clock = 0;
        }

        Bitboard fromBB = BB_SQUARES[move.fromSquare()];
        Bitboard toBB = BB_SQUARES[move.toSquare()];

        bool promoted = (this->promoted & fromBB) != 0;
        auto piece = removePieceAt(move.fromSquare());
        assert(piece.has_value()); // Move must be at least pseudo-legal
        auto pieceType = piece.value().pieceType();

        Square captureSquare = move.toSquare();
        auto capturedPieceType = pieceTypeAt(captureSquare);

        // Update castling rights
        castling_rights &= ~fromBB & ~toBB;
        if (piece.value().pieceType() == PieceType::KING && !promoted) {
            if (turn == WHITE) {
                castling_rights &= ~BB_RANK_1;
            } else {
                castling_rights &= ~BB_RANK_8;
            }
        } else if (capturedPieceType == PieceType::KING && !(this->promoted & toBB)) {
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
                       (std::abs(distance) == 7 || std::abs(distance) == 9)) {
                // En passant capture
                int down = (turn == WHITE) ? -8 : 8;
                captureSquare = move.toSquare() + down;
                capturedPieceType = _removePieceAt(captureSquare);
            }
        }

        // Handle promotions
        if (move.promotion().has_value()) {
            promoted = true;
            pieceType = move.promotion().value();
        }

        // Handle castling moves
        if (pieceType == PieceType::KING && occupied_co[turn] & toBB) {
            // Is a castling move

            bool aSide = squareFile(move.toSquare()) < squareFile(move.fromSquare());

            _removePieceAt(move.toSquare());
            _removePieceAt(move.fromSquare());

            if (aSide) {
                _setPieceAt((turn == WHITE) ? D1 : D8, PieceType::ROOK, turn);
                _setPieceAt((turn == WHITE) ? C1 : C8, PieceType::KING, turn);
            } else {
                _setPieceAt((turn == WHITE) ? F1 : F8, PieceType::ROOK, turn);
                _setPieceAt((turn == WHITE) ? G1 : G8, PieceType::KING, turn);
            }

        } else {
            // Is not a castling move
            _setPieceAt(move.toSquare(), pieceType, turn, promoted);
        }

        // Swap turns
        turn = !turn;
    }

    bool hasPseudoLegalEnPassant() {
        if (!ep_square.has_value())
            return false;

        return !_generatePseudoLegalEP().empty();
    }

    bool hasLegalEnPassant() {
        if (!ep_square.has_value())
            return false;

        return !_generateLegalEP().empty();
    }

    bool isEnPassant(const Move &move) const {
        int distance = std::abs(move.toSquare() - move.fromSquare());
        return ep_square.has_value() && move.toSquare() == ep_square.value() &&
               (pawns & BB_SQUARES[move.fromSquare()]) && (distance == 7 || distance == 9) &&
               !(occupied & BB_SQUARES[move.toSquare()]);
    }

    bool isCapture(const Move &move) const {
        Bitboard touched = BB_SQUARES[move.fromSquare()] ^ BB_SQUARES[move.toSquare()];
        return (touched & occupied_co[!turn]) || isEnPassant(move);
    }

    bool isZeroing(const Move &move) const {
        Bitboard touched = BB_SQUARES[move.fromSquare()] ^ BB_SQUARES[move.toSquare()];
        return (touched & pawns) || (touched & occupied_co[!turn]) ||
               (move.promotion().has_value() && move.promotion() == PieceType::PAWN);
    }

    bool reducesCastlingRights(const Move &move) const {
        Bitboard cr = _cleanCastlingRights();
        Bitboard touched = BB_SQUARES[move.fromSquare()] ^ BB_SQUARES[move.toSquare()];
        return (touched & cr) ||
               (cr & BB_RANK_1 && touched & kings & occupied_co[WHITE] & ~promoted) ||
               (cr & BB_RANK_8 && touched & kings & occupied_co[BLACK] & ~promoted);
    }

    bool isCastling(const Move &move) const {
        if (kings & BB_SQUARES[move.fromSquare()]) {
            int diff = squareFile(move.fromSquare()) - squareFile(move.toSquare());
            return std::abs(diff) > 1 || (rooks & occupied_co[turn] & BB_SQUARES[move.toSquare()]);
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
        Bitboard kingMask = kings & occupied_co[color] & back_rank & ~promoted;
        if (kingMask == BB_EMPTY) {
            return false;
        }

        Bitboard castlingRights = _cleanCastlingRights() & back_rank;
        while (castlingRights) {
            // Isolate the least-significant bit (rightmost rook)
            Bitboard rook = castlingRights & -castlingRights;

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
        Bitboard kingMask = kings & occupied_co[color] & back_rank & ~promoted;
        if (kingMask == BB_EMPTY) {
            return false;
        }

        Bitboard castlingRights = _cleanCastlingRights() & back_rank;
        while (castlingRights) {
            // Isolate the least-significant bit (leftmost rook)
            Bitboard rook = castlingRights & -castlingRights;

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

        if (!occupied) {
            errors |= Status::EMPTY;
        }

        // Kings
        if (!(occupied_co[WHITE] & kings)) {
            errors |= Status::NO_WHITE_KING;
        }
        if (!(occupied_co[BLACK] & kings)) {
            errors |= Status::NO_BLACK_KING;
        }
        if (popcount(occupied & kings) > 2) {
            errors |= Status::TOO_MANY_KINGS;
        }

        // Piece counts
        if (popcount(occupied_co[WHITE]) > 16) {
            errors |= Status::TOO_MANY_WHITE_PIECES;
        }
        if (popcount(occupied_co[BLACK]) > 16) {
            errors |= Status::TOO_MANY_BLACK_PIECES;
        }

        // Pawn counts and positions
        if (popcount(occupied_co[WHITE] & pawns) > 8) {
            errors |= Status::TOO_MANY_WHITE_PAWNS;
        }
        if (popcount(occupied_co[BLACK] & pawns) > 8) {
            errors |= Status::TOO_MANY_BLACK_PAWNS;
        }
        if (pawns & BB_BACK_RANKS) {
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
        Bitboard our_kings = kings & occupied_co[turn] & ~promoted;
        if (checkers && valid_ep_square != std::nullopt) {
            Bitboard pushed_to = valid_ep_square.value() ^ (turn == WHITE ? A2 : A7);
            Bitboard pushed_from = valid_ep_square.value() ^ (turn == WHITE ? A4 : A5);
            Bitboard occupied_before =
                (occupied & ~BB_SQUARES[pushed_to]) | BB_SQUARES[pushed_from];
            if (popcount(checkers) > 1 ||
                (msb(checkers) != pushed_to && _attackedForKing(our_kings, occupied_before))) {
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
        if (!(pawns & occupied_co[!turn] & pawn_mask))
            return std::nullopt;
        if (occupied & BB_SQUARES[ep_square.value()])
            return std::nullopt;
        if (occupied & seventh_rank_mask)
            return std::nullopt;

        return ep_square;
    }

    bool isValid() const { return status() == Status::VALID; }

    bool epSkewered(Square king, Square capturer) const {
        assert(ep_square.has_value());

        Square last_double = ep_square.value() + ((turn == WHITE) ? -8 : 8);
        Bitboard occupancy = occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] |
                             BB_SQUARES[ep_square.value()];

        Bitboard horizontal_attackers = occupied_co[!turn] & (rooks | queens);
        if (BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & occupancy] & horizontal_attackers) {
            return true;
        }

        Bitboard diagonal_attackers = occupied_co[!turn] & (bishops | queens);
        if (BB_DIAG_ATTACKS[king][BB_DIAG_MASKS[king] & occupancy] & diagonal_attackers) {
            return true;
        }

        return false;
    }

    Bitboard sliderBlockers(Square king) const {
        Bitboard rooks_and_queens = rooks | queens;
        Bitboard bishops_and_queens = bishops | queens;

        Bitboard snipers = ((BB_RANK_ATTACKS[king][0] & rooks_and_queens) |
                            (BB_FILE_ATTACKS[king][0] & rooks_and_queens) |
                            (BB_DIAG_ATTACKS[king][0] & bishops_and_queens)) &
                           occupied_co[!turn];

        Bitboard blockers = BB_EMPTY;
        for (Square sniper : scanReversed(snipers)) {
            Bitboard b = between(king, sniper) & occupied;
            if (b && popcount(b) == 1) {
                blockers |= b;
            }
        }

        return blockers & occupied_co[turn];
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

    bool operator==(Board &other) {
        return halfmove_clock == other.halfmove_clock && fullmove_number == other.fullmove_number &&
               pawns == other.pawns && knights == other.knights && bishops == other.bishops &&
               rooks == other.rooks && queens == other.queens && kings == other.kings &&
               occupied == other.occupied && occupied_co == other.occupied_co &&
               promoted == other.promoted && _transpositionKey() == other._transpositionKey();
    }

    Board copy() const {
        // Creates a copy of the board
        Board board(false);

        board.ep_square = ep_square;
        board.castling_rights = castling_rights;
        board.halfmove_clock = halfmove_clock;
        board.fullmove_number = fullmove_number;
        board.turn = turn;

        board.pawns = pawns;
        board.knights = knights;
        board.bishops = bishops;
        board.rooks = rooks;
        board.queens = queens;
        board.kings = kings;
        board.occupied = occupied;
        board.occupied_co = occupied_co;
        board.promoted = promoted;

        return board;
    }

private:
    Bitboard _attacksMask(Square square) const {
        Bitboard bb_square = BB_SQUARES[square];

        if (pawns & bb_square) {
            Color color = (occupied_co[WHITE] & bb_square) ? WHITE : BLACK;
            return BB_PAWN_ATTACKS[color][square];
        }
        if (knights & bb_square) {
            return BB_KNIGHT_ATTACKS[square];
        }
        if (kings & bb_square) {
            return BB_KING_ATTACKS[square];
        }

        Bitboard attacks = 0;
        if (bishops & bb_square || queens & bb_square) {
            attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied];
        }
        if (rooks & bb_square || queens & bb_square) {
            attacks |= (BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] |
                        BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied]);
        }
        return attacks;
    }

    Bitboard _attackersMask(Color color, Square square) const {
        Bitboard rank_pieces = BB_RANK_MASKS[square] & occupied;
        Bitboard file_pieces = BB_FILE_MASKS[square] & occupied;
        Bitboard diag_pieces = BB_DIAG_MASKS[square] & occupied;

        Bitboard queens_and_rooks = queens | rooks;
        Bitboard queens_and_bishops = queens | bishops;

        Bitboard attackers =
            ((BB_KING_ATTACKS[square] & kings) | (BB_KNIGHT_ATTACKS[square] & knights) |
             (BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks) |
             (BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks) |
             (BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops) |
             (BB_PAWN_ATTACKS[!color][square] & pawns));

        return attackers & occupied_co[color];
    }

    Bitboard _pinMask(Color color, Square square) const {
        auto king = this->king(color);
        if (!king.has_value()) {
            return BB_ALL;
        }

        Bitboard square_mask = BB_SQUARES[square];
        Bitboard pinMask = BB_ALL;

        std::array<std::pair<std::array<std::unordered_map<Bitboard, Bitboard>, 64> &, Bitboard>, 3>
            attacks_and_sliders = {{{BB_FILE_ATTACKS, rooks | queens},
                                    {BB_RANK_ATTACKS, rooks | queens},
                                    {BB_DIAG_ATTACKS, bishops | queens}}};

        for (auto &[attacks, sliders] : attacks_and_sliders) {
            Bitboard rays = attacks[king.value()][0];
            if (rays & square_mask) {
                Bitboard snipers = rays & sliders & occupied_co[!color];
                for (auto sniper : scanReversed(snipers)) {
                    if (between(sniper, king.value()) & (occupied | square_mask) == square_mask) {
                        pinMask = ray(king.value(), sniper);
                        break;
                    }
                }
            }
        }

        return pinMask;
    }

    std::optional<PieceType> _removePieceAt(Square square) {
        auto piece_type = pieceTypeAt(square);
        Bitboard mask = BB_SQUARES[square];

        if (piece_type.has_value()) {
            switch (piece_type.value()) {
            case PieceType::PAWN:
                pawns ^= mask;
                break;
            case PieceType::KNIGHT:
                knights ^= mask;
                break;
            case PieceType::BISHOP:
                bishops ^= mask;
                break;
            case PieceType::ROOK:
                rooks ^= mask;
                break;
            case PieceType::QUEEN:
                queens ^= mask;
                break;
            case PieceType::KING:
                kings ^= mask;
                break;
            default:
                return std::nullopt;
            }

            occupied ^= mask;
            occupied_co[WHITE] &= ~mask;
            occupied_co[BLACK] &= ~mask;
            promoted &= ~mask;
        }

        return piece_type;
    }

    void _setPieceAt(Square square, PieceType piece_type, Color color, bool promoted = false) {
        removePieceAt(square); // Clear the square first

        Bitboard mask = BB_SQUARES[square];

        switch (piece_type) {
        case PieceType::PAWN:
            pawns |= mask;
            break;
        case PieceType::KNIGHT:
            knights |= mask;
            break;
        case PieceType::BISHOP:
            bishops |= mask;
            break;
        case PieceType::ROOK:
            rooks |= mask;
            break;
        case PieceType::QUEEN:
            queens |= mask;
            break;
        case PieceType::KING:
            kings |= mask;
            break;
        }

        occupied |= mask;
        occupied_co[color] |= mask;

        if (promoted) {
            this->promoted |= mask;
        }
    }

    std::vector<Move> _generatePseudoLegalMoves(Bitboard from_mask = BB_ALL,
                                                Bitboard to_mask = BB_ALL) {
        if (cachedPseudoLegalMoves.has_value()) {
            return cachedPseudoLegalMoves.value();
        }

        std::vector<Move> moves;
        Bitboard our_pieces = occupied_co[turn];

        // Generate piece moves for non-pawn pieces
        Bitboard non_pawns = our_pieces & ~pawns & from_mask;
        for (Square from_square : scanReversed(non_pawns)) {
            Bitboard move_targets = _attacksMask(from_square) & ~our_pieces & to_mask;
            for (int to_square : scanReversed(move_targets)) {
                moves.push_back(Move(from_square, to_square));
            }
        }

        // Generate castling moves

        if (from_mask & kings) {
            auto castlingMoves = _generateCastlingMoves(from_mask, to_mask);
            moves.insert(moves.end(), castlingMoves.begin(), castlingMoves.end());
        }

        // Generate pawn moves
        Bitboard pawns = this->pawns & occupied_co[turn] & from_mask;
        // Generate pawn captures
        for (int from_square : scanReversed(pawns)) {
            Bitboard targets = BB_PAWN_ATTACKS[turn][from_square] & occupied_co[!turn] & to_mask;
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
            single_moves = shiftUp(pawns) & ~occupied;
            double_moves = shiftUp(single_moves) & ~occupied & (BB_RANK_3 | BB_RANK_4);
        } else {
            single_moves = shiftDown(pawns) & ~occupied;
            double_moves = shiftDown(single_moves) & ~occupied & (BB_RANK_6 | BB_RANK_5);
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
            moves.insert(moves.end(), epMoves.begin(), epMoves.end());
        }

        this->cachedPseudoLegalMoves = moves;
        return moves;
    }

    std::vector<Move> _generatePseudoLegalEP(Bitboard from_mask = BB_ALL,
                                             Bitboard to_mask = BB_ALL) {
        if (cachedPseudoLegalEPMoves.has_value()) {
            return cachedPseudoLegalEPMoves.value();
        }

        std::vector<Move> moves;
        if (!ep_square.has_value() || !(BB_SQUARES[ep_square.value()] & to_mask) ||
            (BB_SQUARES[ep_square.value()] & occupied)) {
            return moves;
        }

        Bitboard capturers = pawns & occupied_co[turn] & from_mask &
                             BB_PAWN_ATTACKS[!turn][ep_square.value()] & BB_RANKS[turn ? 4 : 3];
        for (int capturer : scanReversed(capturers)) {
            moves.emplace_back(capturer, ep_square.value());
        }

        this->cachedPseudoLegalEPMoves = moves;
        return moves;
    }

    Bitboard _checkersMask() const {
        auto king = this->king(turn);
        if (!king.has_value())
            return BB_EMPTY;
        return _attackersMask(!turn, king.value());
    }

    bool _wasIntoCheck() const {
        auto king = this->king(!turn);
        if (!king.has_value())
            return false;
        return isAttackedBy(turn, king.value());
    }

    Bitboard _cleanCastlingRights() const {
        Bitboard castling = castling_rights & rooks;
        Bitboard white_castling = castling & BB_RANK_1 & occupied_co[WHITE];
        Bitboard black_castling = castling & BB_RANK_8 & occupied_co[BLACK];

        white_castling &= BB_A1 | BB_H1;
        black_castling &= BB_A8 | BB_H8;

        if (!(occupied_co[WHITE] & kings & ~promoted & BB_E1))
            white_castling = BB_EMPTY;
        if (!(occupied_co[BLACK] & kings & ~promoted & BB_E8))
            black_castling = BB_EMPTY;

        return white_castling | black_castling;
    }

    bool _isHalfMoves(int n) { return halfmove_clock >= n && !_generateLegalMoves().empty(); }

    Move _findMove(int from_square, int to_square,
                   std::optional<PieceType> promotion = std::nullopt) {
        if (!promotion.has_value() && (pawns & BB_SQUARES[from_square]) &&
            (BB_SQUARES[to_square] & BB_BACK_RANKS)) {
            promotion = PieceType::QUEEN;
        }

        Move move(from_square, to_square, promotion);
        if (!isLegal(move)) {
            throw IllegalMoveError("No matching legal move found in the current position.");
        }

        return move;
    }

    std::vector<Move> _generateEvasions(Square king, Bitboard checkers, Bitboard from_mask = BB_ALL,
                                        Bitboard to_mask = BB_ALL) {
        std::vector<Move> evasions;
        Bitboard sliders = checkers & (bishops | rooks | queens);

        Bitboard attacked = 0;
        for (int checker : scanReversed(sliders)) {
            attacked |= ray(king, checker) & ~BB_SQUARES[checker];
        }

        if (BB_SQUARES[king] & from_mask) {
            for (int to_square :
                 scanReversed(BB_KING_ATTACKS[king] & ~occupied_co[turn] & ~attacked & to_mask)) {
                evasions.emplace_back(king, to_square);
            }
        }

        int checker = msb(checkers);
        if (BB_SQUARES[checker] == checkers) {
            // Capture or block a single checker.
            Bitboard target = between(king, checker) | checkers;
            auto directEvasions = _generatePseudoLegalMoves(~kings & from_mask, target & to_mask);
            evasions.insert(evasions.end(), directEvasions.begin(), directEvasions.end());

            // Capture the checking pawn en passant (avoiding duplicates).
            if (ep_square.has_value() && !(BB_SQUARES[ep_square.value()] & target)) {
                int last_double = ep_square.value() + (turn == WHITE ? -8 : 8);
                if (last_double == checker) {
                    auto epEvasions = _generatePseudoLegalEP(from_mask, to_mask);
                    evasions.insert(evasions.end(), epEvasions.begin(), epEvasions.end());
                }
            }
        }

        return evasions;
    }

    std::vector<Move> _generateLegalMoves(Bitboard from_mask = BB_ALL, Bitboard to_mask = BB_ALL) {
        if (cachedLegalMoves.has_value()) {
            return cachedLegalMoves.value();
        }

        std::vector<Move> legalMoves;
        Bitboard king_mask = kings & occupied_co[turn];
        if (king_mask) {
            Square king = (Square) msb(king_mask);
            Bitboard blockers = sliderBlockers(king);
            Bitboard checkers = _attackersMask(!turn, king);
            if (checkers) {
                auto evasions = _generateEvasions(king, checkers, from_mask, to_mask);
                for (const Move &move : evasions) {
                    if (isSafe(king, blockers, move)) {
                        legalMoves.push_back(move);
                    }
                }
            } else {
                auto pseudoLegalMoves = _generatePseudoLegalMoves(from_mask, to_mask);
                for (const Move &move : pseudoLegalMoves) {
                    if (isSafe(king, blockers, move)) {
                        legalMoves.push_back(move);
                    }
                }
            }
        } else {
            auto pseudoLegalMoves = _generatePseudoLegalMoves(from_mask, to_mask);
            std::copy(pseudoLegalMoves.begin(), pseudoLegalMoves.end(),
                      std::back_inserter(legalMoves));
        }

        this->cachedLegalMoves = legalMoves;
        return legalMoves;
    }

    std::vector<Move> _generateLegalEP(Bitboard from_mask = BB_ALL, Bitboard to_mask = BB_ALL) {
        if (cachedLegalEPMoves.has_value()) {
            return cachedLegalEPMoves.value();
        }

        std::vector<Move> legalEP;
        auto pseudoLegalEP = _generatePseudoLegalEP(from_mask, to_mask);
        for (const Move &move : pseudoLegalEP) {
            if (!isIntoCheck(move)) {
                legalEP.push_back(move);
            }
        }

        this->cachedLegalEPMoves = legalEP;
        return legalEP;
    }

    bool _attackedForKing(Bitboard path, Bitboard occupied) const {
        for (int sq = msb(path); path; sq = msb(path)) {
            if (_attackersMask(!turn, (Square) sq)) {
                return true;
            }
            path &= path - 1; // Clear the scanned bit
        }
        return false;
    }

    std::vector<Move> _generateCastlingMoves(Bitboard from_mask = BB_ALL,
                                             Bitboard to_mask = BB_ALL) const {
        std::vector<Move> moves;

        Bitboard back_rank = (turn == WHITE) ? BB_RANK_1 : BB_RANK_8;
        Bitboard king = occupied_co[turn] & kings & ~promoted & back_rank & from_mask;
        if (!king) {
            return moves;
        }

        Bitboard bb_c = BB_FILE_C & back_rank;
        Bitboard bb_d = BB_FILE_D & back_rank;
        Bitboard bb_f = BB_FILE_F & back_rank;
        Bitboard bb_g = BB_FILE_G & back_rank;

        Bitboard clean_rights = _cleanCastlingRights();
        for (int candidate = msb(clean_rights & back_rank & to_mask); clean_rights;
             candidate = msb(clean_rights)) {
            Bitboard rook = BB_SQUARES[candidate];

            bool a_side = rook < king;
            Bitboard king_to = a_side ? bb_c : bb_g;
            Bitboard rook_to = a_side ? bb_d : bb_f;

            Bitboard king_path = between(msb(king), msb(king_to));
            Bitboard rook_path = between(candidate, msb(rook_to));

            if (!((occupied ^ king ^ rook) & (king_path | rook_path | king_to | rook_to)) &&
                !_attackedForKing(king_path | king, occupied ^ king) &&
                !_attackedForKing(king_to, occupied ^ king ^ rook ^ rook_to)) {
                moves.emplace_back(candidate, msb(king_to));
            }
            clean_rights &= clean_rights - 1; // Clear the scanned bit
        }
        return moves;
    }

    using TranspositionKey = std::tuple<Bitboard, Bitboard, Bitboard, Bitboard, Bitboard, Bitboard,
                                        Bitboard, Bitboard, Color, Bitboard, std::optional<Square>>;

    TranspositionKey _transpositionKey() {
        return {pawns,
                knights,
                bishops,
                rooks,
                queens,
                kings,
                occupied_co[WHITE],
                occupied_co[BLACK],
                turn,
                _cleanCastlingRights(),
                hasLegalEnPassant() ? ep_square : std::optional<Square>{}};
    }
};
} // namespace chess
