#pragma once

#include <array>
#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// Namespace declaration (optional)
namespace chess {
enum Color : bool { WHITE, BLACK };
Color operator!(Color color) { return color == Color::WHITE ? Color::BLACK : Color::WHITE; }

constexpr std::array<Color, 2> COLORS = {Color::WHITE, Color::BLACK};
const std::array<std::string, 2> COLOR_NAMES = {"black", "white"};

enum class PieceType { NONE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
constexpr std::array<PieceType, 7> PIECE_TYPES = {PieceType::PAWN,   PieceType::KNIGHT,
                                                  PieceType::BISHOP, PieceType::ROOK,
                                                  PieceType::QUEEN,  PieceType::KING};
const std::array<std::string, 7> PIECE_SYMBOLS = {" ", "p", "n", "b", "r", "q", "k"};
const std::array<std::string, 7> PIECE_NAMES = {"none", "pawn",  "knight", "bishop",
                                                "rook", "queen", "king"};
std::string piece_symbol(PieceType piece_type) {
    return PIECE_SYMBOLS[static_cast<int>(piece_type)];
}

std::string piece_name(PieceType piece_type) { return PIECE_NAMES[static_cast<int>(piece_type)]; }
const std::unordered_map<char, std::string> UNICODE_PIECE_SYMBOLS = {
    {'R', "♖"}, {'r', "♜"}, {'N', "♘"}, {'n', "♞"}, {'B', "♗"}, {'b', "♝"},
    {'Q', "♕"}, {'q', "♛"}, {'K', "♔"}, {'k', "♚"}, {'P', "♙"}, {'p', "♟"},
};
const std::array<std::string, 8> FILE_NAMES = {"a", "b", "c", "d", "e", "f", "g", "h"};
const std::array<std::string, 8> RANK_NAMES = {"1", "2", "3", "4", "5", "6", "7", "8"};

// The FEN for the standard chess starting position.
const std::string STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// The board part of the FEN for the standard chess starting position.
const std::string STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";

enum class Status : int {
    VALID = 0,
    NO_WHITE_KING = 1 << 0,
    NO_BLACK_KING = 1 << 1,
    TOO_MANY_KINGS = 1 << 2,
    TOO_MANY_WHITE_PAWNS = 1 << 3,
    TOO_MANY_BLACK_PAWNS = 1 << 4,
    PAWNS_ON_BACKRANK = 1 << 5,
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

enum class Termination {
    CHECKMATE,
    // See :func:`chess.Board.is_checkmate()`.
    STALEMATE,
    // See :func:`chess.Board.is_stalemate()`.
    INSUFFICIENT_MATERIAL,
    // See :func:`chess.Board.is_insufficient_material()`.
    SEVENTYFIVE_MOVES,
    // See :func:`chess.Board.is_seventyfive_moves()`.
    FIVEFOLD_REPETITION,
    // See :func:`chess.Board.is_fivefold_repetition()`.
    FIFTY_MOVES,
    // See :func:`chess.Board.can_claim_fifty_moves()`.
    THREEFOLD_REPETITION,
    // See :func:`chess.Board.can_claim_threefold_repetition()`.
    VARIANT_WIN,
    // See :func:`chess.Board.is_variant_win()`.
    VARIANT_LOSS,
    // See :func:`chess.Board.is_variant_loss()`.
    VARIANT_DRAW,
    // See :func:`chess.Board.is_variant_draw()`.
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

class AmbiguousMoveError : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// clang-format off
enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1, //
    A2, B2, C2, D2, E2, F2, G2, H2, //
    A3, B3, C3, D3, E3, F3, G3, H3, //
    A4, B4, C4, D4, E4, F4, G4, H4, //
    A5, B5, C5, D5, E5, F5, G5, H5, //
    A6, B6, C6, D6, E6, F6, G6, H6, //
    A7, B7, C7, D7, E7, F7, G7, H7, //
    A8, B8, C8, D8, E8, F8, G8, H8, //
};

constexpr int operator+(Square square) { return static_cast<int>(square); }


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

const  std::array<std::string, 64> SQUARE_NAMES = {
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

int parse_square(const std::string &name) {
    auto it = std::find(SQUARE_NAMES.begin(), SQUARE_NAMES.end(), name);
    if (it == SQUARE_NAMES.end()) {
        throw std::invalid_argument("Invalid square name");
    }
    return std::distance(SQUARE_NAMES.begin(), it);
}

std::string square_name(Square square) {
    if (square < 0 || square >= 64) {
        throw std::invalid_argument("Square index out of bounds");
    }
    return SQUARE_NAMES[square];
}

Square square(int file_index, int rank_index) { return (Square) (rank_index * 8 + file_index); }

Square square_file(Square square) { return (Square) (square & 7); }

Square square_rank(Square square) { return (Square) (square >> 3); }

int square_distance(Square a, Square b) {
    return std::max(std::abs(square_file(a) - square_file(b)),
                    std::abs(square_rank(a) - square_rank(b)));
}

int square_manhattan_distance(Square a, Square b) {
    return std::abs(square_file(a) - square_file(b)) + std::abs(square_rank(a) - square_rank(b));
}

int square_knight_distance(Square a, Square b) {
    int dx = std::abs(square_file(a) - square_file(b));
    int dy = std::abs(square_rank(a) - square_rank(b));

    if (dx + dy == 1) {
        return 3;
    } else if (dx == 2 && dy == 2) {
        return 4;
    } else if (dx == 1 && dy == 1) {
        // Assuming BB_SQUARES and BB_CORNERS are bitboards representing squares and corners
        // This part of the logic depends on the representation of BB_SQUARES and BB_CORNERS
        // which are not provided in the Python code snippet.
        // You'll need to adjust this part based on your actual bitboard implementation.
        throw std::runtime_error("Bitboard operations for knight distance not implemented");
    }

    int m = std::ceil(std::max({dx / 2.0, dy / 2.0, (dx + dy) / 3.0}));
    return m + ((m + dx + dy) % 2);
}

constexpr Square square_mirror(Square square) { return (Square) (square ^ 0x38); }

constexpr std::array<Square, 64> SQUARES_180 = [] {
    std::array<Square, 64> squares_180;
    for (size_t i = 0; i < SQUARES.size(); ++i) {
        squares_180[i] = square_mirror(SQUARES[i]);
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

constexpr Bitboard BB_CORNERS = BB_SQUARES[A1] | BB_SQUARES[H1] | BB_SQUARES[A8] | BB_SQUARES[H8];
constexpr Bitboard BB_CENTER = BB_SQUARES[D4] | BB_SQUARES[E4] | BB_SQUARES[D5] | BB_SQUARES[E5];

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

constexpr Bitboard BB_BACKRANKS = BB_RANKS_1 | BB_RANKS_8;

int lsb(Bitboard bb) {
    if (bb == 0)
        return -1;
    return __builtin_ctzll(bb);
}

std::vector<int> scan_forward(Bitboard bb) {
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
    return 63 - __builtin_clzll(bb);
}

std::vector<Square> scan_reversed(Bitboard bb) {
    std::vector<Square> squares;
    while (bb) {
        int sq = msb(bb);
        squares.push_back((Square) sq);
        bb &= ~(1ULL << sq); // Reset MSB
    }
    return squares;
}

int popcount(Bitboard bb) { return __builtin_popcountll(bb); }

Bitboard flip_vertical(Bitboard bb) {
    bb = ((bb >> 8) & 0x00FF'00FF'00FF'00FF) | ((bb & 0x00FF'00FF'00FF'00FF) << 8);
    bb = ((bb >> 16) & 0x0000'FFFF'0000'FFFF) | ((bb & 0x0000'FFFF'0000'FFFF) << 16);
    bb = (bb >> 32) | ((bb & 0x0000'0000'FFFF'FFFF) << 32);
    return bb;
}

Bitboard flip_horizontal(Bitboard bb) {
    bb = ((bb >> 1) & 0x5555'5555'5555'5555) | ((bb & 0x5555'5555'5555'5555) << 1);
    bb = ((bb >> 2) & 0x3333'3333'3333'3333) | ((bb & 0x3333'3333'3333'3333) << 2);
    bb = ((bb >> 4) & 0x0F0F'0F0F'0F0F'0F0F) | ((bb & 0x0F0F'0F0F'0F0F'0F0F) << 4);
    return bb;
}

Bitboard flip_diagonal(Bitboard bb) {
    Bitboard t;
    t = (bb ^ (bb << 28)) & 0x0F0F'0F0F'0000'0000ULL;
    bb = bb ^ t ^ (t >> 28);
    t = (bb ^ (bb << 14)) & 0x3333'0000'3333'0000ULL;
    bb = bb ^ t ^ (t >> 14);
    t = (bb ^ (bb << 7)) & 0x5500'5500'5500'5500ULL;
    bb = bb ^ t ^ (t >> 7);
    return bb;
}

Bitboard flip_anti_diagonal(Bitboard bb) {
    Bitboard t;
    t = bb ^ (bb << 36);
    bb = bb ^ ((t ^ (bb >> 36)) & 0xF0F0'F0F0'0F0F'0F0FULL);
    t = (bb ^ (bb << 18)) & 0xCCCC'0000'CCCC'0000ULL;
    bb = bb ^ t ^ (t >> 18);
    t = (bb ^ (bb << 9)) & 0xAA00'AA00'AA00'AA00ULL;
    bb = bb ^ t ^ (t >> 9);
    return bb;
}

Bitboard shift_down(Bitboard b) { return b >> 8; }

Bitboard shift_2_down(Bitboard b) { return b >> 16; }

Bitboard shift_up(Bitboard b) { return (b << 8) & BB_ALL; }

Bitboard shift_2_up(Bitboard b) { return (b << 16) & BB_ALL; }

Bitboard shift_right(Bitboard b) { return (b << 1) & ~BB_FILES[0] & BB_ALL; }

Bitboard shift_2_right(Bitboard b) { return (b << 2) & ~BB_FILES[0] & ~BB_FILES[1] & BB_ALL; }

Bitboard shift_left(Bitboard b) { return (b >> 1) & ~BB_FILES[7]; }

Bitboard shift_2_left(Bitboard b) { return (b >> 2) & ~BB_FILES[6] & ~BB_FILES[7]; }

Bitboard shift_up_left(Bitboard b) { return (b << 7) & ~BB_FILES[7] & BB_ALL; }

Bitboard shift_up_right(Bitboard b) { return (b << 9) & ~BB_FILES[0] & BB_ALL; }

Bitboard shift_down_left(Bitboard b) { return (b >> 9) & ~BB_FILES[7]; }

Bitboard shift_down_right(Bitboard b) { return (b >> 7) & ~BB_FILES[0]; }

Bitboard _sliding_attacks(Square square, Bitboard occupied, const std::vector<int> &deltas) {
    Bitboard attacks = BB_EMPTY;

    for (auto delta : deltas) {
        int sq = square;

        while (true) {
            sq += delta;
            if (!(0 <= sq && sq < 64) || square_distance((Square) sq, (Square) (sq - delta)) > 2) {
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

Bitboard _step_attacks(Square square, const std::vector<int> &deltas) {
    return _sliding_attacks(square, BB_ALL, deltas);
}

std::array<Bitboard, 64> BB_KNIGHT_ATTACKS = [] {
    std::array<Bitboard, 64> attacks;
    for (Square sq : SQUARES) {
        attacks[sq] = _step_attacks(sq, {17, 15, 10, 6, -17, -15, -10, -6});
    }
    return attacks;
}();

std::array<Bitboard, 64> BB_KING_ATTACKS = [] {
    std::array<Bitboard, 64> attacks;
    for (Square sq : SQUARES) {
        attacks[sq] = _step_attacks(sq, {9, 8, 7, 1, -9, -8, -7, -1});
    }
    return attacks;
}();

std::array<std::array<Bitboard, 64>, 2> BB_PAWN_ATTACKS = [] {
    std::array<std::array<Bitboard, 64>, 2> attacks;
    for (int color = 0; color < 2; ++color) {
        for (Square sq : SQUARES) {
            attacks[color][sq] =
                _step_attacks(sq, color == 0 ? std::vector<int>{-7, -9} : std::vector<int>{7, 9});
        }
    }
    return attacks;
}();

Bitboard _edges(Square square) {
    return (((BB_RANKS[0] | BB_RANKS[7]) & ~BB_RANKS[square_rank(square)]) |
            ((BB_FILES[0] | BB_FILES[7]) & ~BB_FILES[square_file(square)]));
}

std::vector<Bitboard> _carry_rippler(Bitboard mask) {
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

        Bitboard mask = _sliding_attacks(square, 0, deltas) & ~_edges((Square) square);
        for (auto subset : _carry_rippler(mask)) {
            attacks[subset] = _sliding_attacks(square, subset, deltas);
        }

        attack_table[square] = attacks;
        mask_table[square] = mask;
    }

    return {mask_table, attack_table};
}

auto [BB_DIAG_MASKS, BB_DIAG_ATTACKS] = _attack_table({-9, -7, 7, 9});
auto [BB_FILE_MASKS, BB_FILE_ATTACKS] = _attack_table({-8, 8});
auto [BB_RANK_MASKS, BB_RANK_ATTACKS] = _attack_table({-1, 1});

std::vector<std::vector<Bitboard>> _rays() {
    std::vector<std::vector<Bitboard>> rays;
    for (Square a : SQUARES) {
        std::vector<Bitboard> rays_row;
        for (Square b : SQUARES) {
            if (BB_DIAG_ATTACKS[a][0] & BB_SQUARES[b]) {
                rays_row.push_back((BB_DIAG_ATTACKS[a][0] & BB_DIAG_ATTACKS[b][0]) | BB_SQUARES[a] |
                                   BB_SQUARES[b]);
            } else if (BB_RANK_ATTACKS[a][0

            ] & BB_SQUARES[b]) {
                rays_row.push_back(BB_RANK_ATTACKS[a][0] | BB_SQUARES[a]);
            } else if (BB_FILE_ATTACKS[a][0] & BB_SQUARES[b]) {
                rays_row.push_back(BB_FILE_ATTACKS[a][0] | BB_SQUARES[a]);
            } else {
                rays_row.push_back(BB_EMPTY);
            }
        }
        rays.push_back(rays_row);
    }
    return rays;
}

std::vector<std::vector<Bitboard>> BB_RAYS = _rays();

Bitboard ray(int a, int b) { return BB_RAYS[a][b]; }

Bitboard between(int a, int b) {
    Bitboard bb = BB_RAYS[a][b] & ((BB_ALL << a) ^ (BB_ALL << b));
    return bb & (bb - 1);
}

const std::regex
    SAN_REGEX(R"(^([NBKRQ])?([a-h])?([1-8])?[-x]?([a-h][1-8])(=?[nbrqkNBRQK])?[+#]?$)");

const std::regex FEN_CASTLING_REGEX(R"(^(?:-|[KQABCDEFGH]{0,2}[kqabcdefgh]{0,2})$)");

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
        auto symbol = piece_symbol(pieceType());
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
    size_t hash() const { return std::hash<unsigned char>()(value); }

    // Representation of Piece
    std::string repr() const { return "Piece(" + std::string(1, symbol()[0]) + ")"; }

    // String representation
    std::string str() const { return symbol(); }

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

    bool drop() const { return value & DROP_MASK; }

    std::string uci() const {
        if (!*this)
            return "0000";

        std::string uci = SQUARE_NAMES[fromSquare()] + SQUARE_NAMES[toSquare()];
        auto promo = promotion();
        if (promo) {
            uci += piece_symbol(promo.value());
        }
        return uci;
    }

    bool operator!() const { return value == 0; }

    std::string repr() const { return "Move::from_uci(\"" + uci() + "\")"; }

    std::string str() const { return uci(); }

    static Move null() { return Move(); }

    static Move fromUci(const std::string &uci) {
        if (uci == "0000") {
            return null();
        }
        if (uci.size() != 4 && uci.size() != 5) {
            throw std::invalid_argument("Invalid uci string");
        }
        int from = parse_square(uci.substr(0, 2));
        int to = parse_square(uci.substr(2, 2));
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

    // 0000000000001110 for promotion (3 bits)
    static constexpr unsigned short PROMOTION_MASK = 0x000E;
    static constexpr int PROMOTION_SHIFT = 1;

    // 0000000000000001 for drop (1 bit)
    static constexpr unsigned short DROP_MASK = 0x0001;
    static constexpr int DROP_SHIFT = 0;

    int getValue(int from_square, int to_square, std::optional<PieceType> promotion = std::nullopt,
                 bool drop = false) {
        int value = 0;
        value |= (from_square << FROM_SQUARE_SHIFT);
        value |= (to_square << TO_SQUARE_SHIFT);
        if (promotion) {
            value |= (static_cast<unsigned short>(promotion.value()) << PROMOTION_SHIFT);
        }
        value |= (drop << DROP_SHIFT);
        return value;
    }

    const unsigned short value;
};

class BaseBoard {
    friend class _BoardState;

protected:
    std::array<Bitboard, 2> occupied_co; // Bitboards for white and black pieces
    Bitboard pawns, knights, bishops, rooks, queens, kings, promoted, occupied;

public:
    BaseBoard(const std::optional<std::string> &board_fen =
                  std::optional<std::string>(STARTING_BOARD_FEN)) {
        occupied_co = {BB_EMPTY, BB_EMPTY};

        if (!board_fen.has_value()) {
            clearBoard();
        } else if (board_fen == STARTING_BOARD_FEN) {
            resetBoard();
        } else {
            setBoardFen(board_fen.value());
        }
    }

    void resetBoard() {
        // Resets pieces to the starting position.
        //
        // :class:`~chess.Board` also resets the move stack, but not turn,
        // castling rights and move counters. Use :func:`chess.Board.reset()` to
        // fully restore the starting position.

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
        // Clears the board.
        //
        // :class:`~chess.Board` also clears the move stack.

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
        // Returns a :class:`set of squares <chess.SquareSet>`.

        return SquareSet(piecesMask(piece_type, color));
    }

    std::optional<Piece> pieceAt(Square square) const {
        // Gets the :class:`piece <chess.Piece>` at the given square.

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

    Bitboard attacks_mask(Square square) const {
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

    SquareSet attacks(Square square) const {
        // Gets the set of attackers of the given color for the given square.
        //
        // Pinned pieces still count as attackers.
        //
        // Returns a :class:`set of squares <chess.SquareSet>`.

        return SquareSet(attacks_mask(square));
    }

    Bitboard attackersMask(Color color, Square square) const {
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
        // Returns a :class:`set of squares <chess.SquareSet>`.

        return SquareSet(attackersMask(color, square));
    }

    Bitboard pin_mask(Color color, Square square) const {
        auto king = this->king(color);
        if (!king.has_value()) {
            return BB_ALL;
        }

        Bitboard square_mask = BB_SQUARES[square];
        Bitboard pin_mask = BB_ALL;

        std::array<std::pair<std::array<std::unordered_map<Bitboard, Bitboard>, 64> &, Bitboard>, 3>
            attacks_and_sliders = {{{BB_FILE_ATTACKS, rooks | queens},
                                    {BB_RANK_ATTACKS, rooks | queens},
                                    {BB_DIAG_ATTACKS, bishops | queens}}};

        for (auto &[attacks, sliders] : attacks_and_sliders) {
            Bitboard rays = attacks[king.value()][0];
            if (rays & square_mask) {
                Bitboard snipers = rays & sliders & occupied_co[!color];
                for (auto sniper : scan_reversed(snipers)) {
                    if (between(sniper, king.value()) & (occupied | square_mask) == square_mask) {
                        pin_mask = ray(king.value(), sniper);
                        break;
                    }
                }
            }
        }

        return pin_mask;
    }

    SquareSet pin(Color color, Square square) const {
        // Detects an absolute pin (and its direction) of the given square to
        // the king of the given color.
        //
        // >>> import chess
        // >>>
        // >>> board = chess.Board("rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3 7")
        // >>> board.is_pinned(chess.WHITE, chess.C3)
        // True
        // >>> direction = board.pin(chess.WHITE, chess.C3)
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
        // Returns a :class:`set of squares <chess.SquareSet>` that mask the rank,
        // file or diagonal of the pin. If there is no pin, then a mask of the
        // entire board is returned.
        return SquareSet(pin_mask(color, square));
    }

    bool is_pinned(Color color, Square square) const {
        // Detects if the given square is pinned to the king of the given color.
        return pin_mask(color, square) != BB_ALL;
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

    std::optional<Piece> removePieceAt(Square square) {
        auto color = (occupied_co[WHITE] & BB_SQUARES[square]) ? WHITE : BLACK;
        auto piece_type = _removePieceAt(square);
        return piece_type.has_value() ? std::optional<Piece>(Piece(piece_type.value(), color))
                                      : std::nullopt;
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

    void setBoardFen(const std::string &board_fen) {
        // Sets the board to the position represented by the given board FEN string.
        // Also clears the move stack, move counters, and castling rights.
        clearBoard();

        int rank = 7;
        int file = 0;
        for (char c : board_fen) {
            if (c == '/') {
                rank--;
                file = 0;
            } else if (isdigit(c)) {
                file += c - '0';
            } else {
                auto piece = Piece::from_symbol(c);
                setPieceAt(square(file, rank), piece);
                file++;
            }
        }
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

    // Mirrors the board vertically and swaps piece colors.
    void applyMirror() {
        auto mirrorTransform = [](Bitboard bb) -> Bitboard { return flip_vertical(bb); };
        applyTransform(mirrorTransform);
        std::swap(occupied_co[WHITE], occupied_co[BLACK]);
    }

    BaseBoard mirror() {
        BaseBoard mirroredBoard = *this; // Assuming a copy constructor is defined.
        mirroredBoard.applyMirror();
        return mirroredBoard;
    }

    // Applies a transformation function to all bitboards.
    void applyTransform(std::function<Bitboard(Bitboard)> f) {
        pawns = f(pawns);
        knights = f(knights);
        bishops = f(bishops);
        rooks = f(rooks);
        queens = f(queens);
        kings = f(kings);
        occupied = f(occupied);
        promoted = f(promoted);
        for (auto &co : occupied_co) {
            co = f(co);
        }
    }

    bool operator==(const BaseBoard &other) const {
        return pawns == other.pawns && knights == other.knights && bishops == other.bishops &&
               rooks == other.rooks && queens == other.queens && kings == other.kings &&
               occupied == other.occupied && occupied_co == other.occupied_co &&
               promoted == other.promoted;
    }

    BaseBoard copy() const {
        BaseBoard board(std::nullopt);
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
};

class _BoardState {
public:
    Bitboard pawns, knights, bishops, rooks, queens, kings, promoted, occupied;
    Bitboard occupied_w, occupied_b;
    Color turn;
    Bitboard castling_rights;
    std::optional<Square> ep_square;
    int halfmove_clock;
    int fullmove_number;

    // Constructor captures the current state of the board.
    _BoardState(const Board &board)
        : pawns(board.pawns), knights(board.knights), bishops(board.bishops), rooks(board.rooks),
          queens(board.queens), kings(board.kings), promoted(board.promoted),
          occupied(board.occupied), occupied_w(board.occupied_co[WHITE]),
          occupied_b(board.occupied_co[BLACK]), turn(board.turn),
          castling_rights(board.castling_rights), ep_square(board.ep_square),
          halfmove_clock(board.halfmove_clock), fullmove_number(board.fullmove_number) {}

    // Restore the board to the captured state.
    void restore(Board &board) const {
        board.pawns = pawns;
        board.knights = knights;
        board.bishops = bishops;
        board.rooks = rooks;
        board.queens = queens;
        board.kings = kings;
        board.promoted = promoted;
        board.occupied = occupied;

        board.occupied_co[WHITE] = occupied_w;
        board.occupied_co[BLACK] = occupied_b;

        board.turn = turn;
        board.castling_rights = castling_rights;
        board.ep_square = ep_square;
        board.halfmove_clock = halfmove_clock;
        board.fullmove_number = fullmove_number;
    }
};

class Board : public BaseBoard {
    // A :class:`~chess.BaseBoard`, additional information representing
    // a chess position, and a :data:`move stack <chess.Board.move_stack>`.
    //
    // Provides :data:`move generation <chess.Board.legal_moves>`, validation,
    // :func:`parsing <chess.Board.parse_san()>`, attack generation,
    // :func:`game end detection <chess.Board.is_game_over()>`,
    // and the capability to :func:`make <chess.Board.push()>` and
    // :func:`unmake <chess.Board.pop()>` moves.
    //
    // The board is initialized to the standard chess starting position,
    // unless otherwise specified in the optional *fen* argument.
    // If *fen* is ``None``, an empty board is created.
    //
    // It's safe to set :data:`~Board.turn`, :data:`~Board.castling_rights`,
    // :data:`~Board.ep_square`, :data:`~Board.halfmove_clock` and
    // :data:`~Board.fullmove_number` directly.
    //
    // .. warning::
    //     It is possible to set up and work with invalid positions. In this
    //     case, :class:`~chess.Board` implements a kind of "pseudo-chess"
    //     (useful to gracefully handle errors or to implement chess variants).
    //     Use :func:`~chess.Board.is_valid()` to detect invalid positions.

public:
    static inline const std::string uci_variant = "chess";
    static inline const std::string starting_fen = STARTING_FEN;

    Color turn;
    // The side to move (``chess.WHITE`` or ``chess.BLACK``).

    Bitboard castling_rights;
    // Bitmask of the rooks with castling rights.
    //
    // To test for specific squares:
    //
    // >>> import chess
    // >>>
    // >>> board = chess.Board()
    // >>> bool(board.castling_rights & chess.BB_H1)  # White can castle with the h1 rook
    // True
    //
    // To add a specific square:
    //
    // >>> board.castling_rights |= chess.BB_A1
    //
    // Use :func:`~chess.Board.set_castling_fen()` to set multiple castling
    // rights. Also see :func:`~chess.Board.has_castling_rights()`,
    // :func:`~chess.Board.has_kingside_castling_rights()`,
    // :func:`~chess.Board.has_queenside_castling_rights()`,
    // :func:`~chess.Board.clean_castling_rights()`.

    std::optional<Square> ep_square;
    // The potential en passant square on the third or sixth rank or ``None``.
    //
    // Use :func:`~chess.Board.has_legal_en_passant()` to test if en passant
    // capturing would actually be possible on the next move.

    int fullmove_number;
    // Counts move pairs. Starts at `1` and is incremented after every move of the black side.

    int halfmove_clock;
    // The number of half-moves since the last capture or pawn move.

    Bitboard promoted;
    // A bitmask of pieces that have been promoted.

    std::vector<Move> move_stack;
    // The move stack. Use :func:`Board.push() <chess.Board.push()>`,
    // :func:`Board.pop() <chess.Board.pop()>`,
    // :func:`Board.peek() <chess.Board.peek()>` and
    // :func:`Board.clear_stack() <chess.Board.clear_stack()>` for manipulation.

private:
    std::vector<_BoardState> _stack;

public:
    // Constructor
    Board() : BaseBoard(nullptr), ep_square(std::nullopt), fullmove_number(1), halfmove_clock(0) {
        if (!fen.has_value()) {
            clear();
        } else if (fen == starting_fen) {
            reset();
        } else {
            setFen(fen.value());
        }
    }

    LegalMoveGenerator legal_moves() {
        // A dynamic list of legal moves.
        //
        // >>> import chess
        // >>>
        // >>> Board board;
        // >>> board.legal_moves.count()
        // 20
        // >>> (bool)board.legal_moves
        // True
        // >>> Move move = Move.from_uci('g1f3')
        // >>> board.legal_moves.contains(move)
        // True
        //
        // Wraps :func:`~chess.Board.generate_legal_moves()` and
        // :func:`~chess.Board.is_legal()`.

        return LegalMoveGenerator(this);
    }

    PseudoLegalMoveGenerator pseudo_legal_moves() {
        // A dynamic list of pseudo-legal moves, much like the legal move list.
        //
        // Pseudo-legal moves might leave or put the king in check, but are
        // otherwise valid. Null moves are not pseudo-legal. Castling moves are
        // only included if they are completely legal.
        //
        // Wraps :func:`~chess.Board.generate_pseudo_legal_moves()` and
        // :func:`~chess.Board.is_pseudo_legal()`.

        return PseudoLegalMoveGenerator(this);
    }

    // Resets the board to the starting position
    void reset() {
        turn = WHITE;
        castling_rights = BB_CORNERS;
        ep_square = std::nullopt;
        halfmove_clock = 0;
        fullmove_number = 1;

        BaseBoard::resetBoard(); // Call to BaseBoard's resetBoard
        clearStack();
    }

    // Clears the board
    void clear() {
        turn = WHITE;
        castling_rights = BB_EMPTY;
        ep_square = std::nullopt;
        halfmove_clock = 0;
        fullmove_number = 1;

        BaseBoard::clearBoard(); // Call to BaseBoard's clearBoard
        clearStack();
    }

    // Clears the move stack
    void clearStack() {
        move_stack.clear();
        _stack.clear();
    }

    // Returns a copy of the root position
    Board root() {
        if (!_stack.empty()) {
            Board board;
            _stack.front().restore(board);
            return board;
        } else {
            return this->copy();
        }
    }

    // Returns the number of half-moves since the start of the game
    int ply() const { return 2 * (fullmove_number - 1) + (turn == BLACK ? 1 : 0); }

    std::optional<Piece> removePieceAt(Square square) {
        auto piece = BaseBoard::removePieceAt(square);
        clearStack();
        return piece;
    }

    void setPieceAt(Square square, const std::optional<Piece> &piece, bool promoted = false) {
        BaseBoard::setPieceAt(square, piece, promoted);
        clearStack();
    }

    // Assuming the existence of helper functions like scan_reversed and BB_PAWN_ATTACKS setup
    std::vector<Move> generatePseudoLegalMoves(Bitboard from_mask = BB_ALL,
                                               Bitboard to_mask = BB_ALL) {
        std::vector<Move> moves;
        Bitboard our_pieces = occupied_co[turn];

        // Generate piece moves for non-pawn pieces
        Bitboard non_pawns = our_pieces & ~pawns & from_mask;
        for (Square from_square : scan_reversed(non_pawns)) {
            Bitboard move_targets = attacks_mask(from_square) & ~our_pieces & to_mask;
            for (int to_square : scan_reversed(move_targets)) {
                moves.push_back(Move(from_square, to_square));
            }
        }

        // Generate castling moves
        // Assuming generateCastlingMoves returns a std::vector<Move>
        if (from_mask & kings) {
            auto castlingMoves = generateCastlingMoves(from_mask, to_mask);
            moves.insert(moves.end(), castlingMoves.begin(), castlingMoves.end());
        }

        // Generate pawn moves
        Bitboard pawns = this->pawns & occupied_co[turn] & from_mask;
        // Generate pawn captures
        for (int from_square : scan_reversed(pawns)) {
            Bitboard targets = BB_PAWN_ATTACKS[turn][from_square] & occupied_co[!turn] & to_mask;
            for (Square to_square : scan_reversed(targets)) {
                if (square_rank(to_square) in{0, 7}) { // Handle promotions
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
            single_moves = shift_up(pawns) & ~occupied;
            double_moves = shift_up(single_moves) & ~occupied & (BB_RANK_3 | BB_RANK_4);
        } else {
            single_moves = shift_down(pawns) & ~occupied;
            double_moves = shift_down(single_moves) & ~occupied & (BB_RANK_6 | BB_RANK_5);
        }

        single_moves &= to_mask;
        double_moves &= to_mask;

        for (Square to_square : scan_reversed(single_moves)) {
            Square from_square = to_square + (turn == BLACK ? 8 : -8);
            if (square_rank(to_square) in{0, 7}) { // Handle promotions
                moves.push_back(Move(from_square, to_square, PieceType::QUEEN));
                moves.push_back(Move(from_square, to_square, PieceType::ROOK));
                moves.push_back(Move(from_square, to_square, PieceType::BISHOP));
                moves.push_back(Move(from_square, to_square, PieceType::KNIGHT));
            } else {
                moves.push_back(Move(from_square, to_square));
            }
        }

        for (int to_square : scan_reversed(double_moves)) {
            int from_square = to_square + (turn == BLACK ? 16 : -16);
            moves.push_back(Move(from_square, to_square));
        }

        // Generate en passant captures
        if (ep_square.has_value()) {
            auto epMoves = generatePseudoLegalEP(from_mask, to_mask);
            moves.insert(moves.end(), epMoves.begin(), epMoves.end());
        }

        return moves;
    }

    std::vector<Move> generatePseudoLegalEP(Bitboard from_mask = BB_ALL,
                                            Bitboard to_mask = BB_ALL) {
        std::vector<Move> moves;
        if (!ep_square.has_value() || !(BB_SQUARES[ep_square.value()] & to_mask) ||
            (BB_SQUARES[ep_square.value()] & occupied)) {
            return moves;
        }

        Bitboard capturers = pawns & occupied_co[turn] & from_mask &
                             BB_PAWN_ATTACKS[!turn][ep_square.value()] & BB_RANKS[turn ? 4 : 3];
        for (int capturer : scan_reversed(capturers)) {
            moves.emplace_back(capturer, ep_square.value());
        }
        return moves;
    }

    std::vector<Move> generatePseudoLegalCaptures(Bitboard from_mask = BB_ALL,
                                                  Bitboard to_mask = BB_ALL) {
        std::vector<Move> moves;
        auto pieceMoves = generatePseudoLegalMoves(from_mask, to_mask & occupied_co[!turn]);
        auto epMoves = generatePseudoLegalEP(from_mask, to_mask);

        moves.insert(moves.end(), pieceMoves.begin(), pieceMoves.end());
        moves.insert(moves.end(), epMoves.begin(), epMoves.end());

        return moves;
    }

    Bitboard checkersMask() {
        auto king = this->king(turn);
        if (!king.has_value())
            return BB_EMPTY;
        return attackersMask(!turn, king.value());
    }

    bool isCheck() { return checkersMask() != BB_EMPTY; }

    bool givesCheck(const Move &move) {
        push(move);
        bool inCheck = isCheck();
        pop();
        return inCheck;
    }

    bool isIntoCheck(const Move &move) {
        auto king = this->king(turn);
        if (!king.has_value())
            return false;

        Bitboard checkers = attackersMask(!turn, king.value());
        if (checkers) {
            // Assuming _generateEvasions and _isSafe are implemented
            if (std::find(_generateEvasions().begin(), _generateEvasions().end(), move) ==
                _generateEvasions().end()) {
                return true;
            }
        }

        return !_isSafe(king.value(), _sliderBlockers(king.value()), move);
    }

    bool wasIntoCheck() {
        auto king = this->king(!turn);
        if (!king.has_value())
            return false;
        return isAttackedBy(turn, king.value());
    }

    // Check if a move is pseudo-legal
    bool isPseudoLegal(const Move &move) {
        if (!move)
            return false; // Null moves are not pseudo-legal in C++ context

        if (move.drop())
            return false; // Drops are not supported

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
            if (turn == WHITE && square_rank(move.toSquare()) != 7)
                return false;
            if (turn == BLACK && square_rank(move.toSquare()) != 0)
                return false;
        }

        if (piece == PieceType::KING) {
            auto castlingMoves = generateCastlingMoves(); // This needs to be defined
            return std::find(castlingMoves.begin(), castlingMoves.end(), move) !=
                   castlingMoves.end();
        }

        if (occupied_co[turn] & to_mask)
            return false; // Destination square cannot be occupied by our piece

        if (piece == PieceType::PAWN) {
            auto pawnMoves =
                generatePseudoLegalMoves(from_mask, to_mask); // Specific pawn moves handling
            return std::find(pawnMoves.begin(), pawnMoves.end(), move) != pawnMoves.end();
        }

        return attacks_mask(move.fromSquare()) & to_mask;
    }

    // Check if a move is legal
    bool isLegal(const Move &move) {
        return isPseudoLegal(move) && !isIntoCheck(move); // Assuming isIntoCheck is defined
    }

    // Check if the game is over
    bool isGameOver(bool claim_draw = false) { return outcome(claim_draw).has_value(); }

    // Get the game result as a string
    std::string result(bool claim_draw = false) {
        auto outcomeOption = outcome(claim_draw);
        return outcomeOption ? outcomeOption->result() : "*";
    }

    // Determine the outcome of the game
    std::optional<Outcome> outcome(bool claim_draw = false) {
        if (isCheckmate())
            return Outcome(Termination::CHECKMATE, !turn);
        if (isInsufficientMaterial())
            return Outcome(Termination::INSUFFICIENT_MATERIAL, std::nullopt);
        if (!any(generateLegalMoves()))
            return Outcome(Termination::STALEMATE, std::nullopt);

        if (isSeventyFiveMoves())
            return Outcome(Termination::SEVENTYFIVE_MOVES, std::nullopt);
        if (isFivefoldRepetition())
            return Outcome(Termination::FIVEFOLD_REPETITION, std::nullopt);

        if (claim_draw) {
            if (canClaimFiftyMoves())
                return Outcome(Termination::FIFTY_MOVES, std::nullopt);
            if (canClaimThreefoldRepetition())
                return Outcome(Termination::THREEFOLD_REPETITION, std::nullopt);
        }

        return std::nullopt; // No outcome determined
    }

    bool isCheckmate() {
        if (!isCheck())
            return false;

        return generateLegalMoves().empty();
    }

    bool isStalemate() {
        if (isCheck())
            return false;

        if (isVariantEnd())
            return false; // Implement isVariantEnd() based on specific variant rules

        return generateLegalMoves().empty();
    }

    bool isInsufficientMaterial() {
        for (Color color : {WHITE, BLACK}) {
            if (!hasInsufficientMaterial(color))
                return false;
        }
        return true;
    }

    bool hasInsufficientMaterial(Color color) {
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

    bool _isHalfMoves(int n) { return halfmove_clock >= n && !generateLegalMoves().empty(); }

    bool isSeventyFiveMoves() { return _isHalfMoves(150); }

    bool isFivefoldRepetition() {
        // Implement this based on your repetition detection logic
        return isRepetition(5);
    }

    bool canClaimDraw() {
        return canClaimFiftyMoves() ||
               canClaimThreefoldRepetition(); // Implement canClaimThreefoldRepetition based on your
                                              // repetition detection logic
    }

    bool isFiftyMoves() { return _isHalfMoves(100); }

    bool canClaimFiftyMoves() {
        if (isFiftyMoves())
            return true;

        if (halfmove_clock >= 99) {
            for (const Move &move : generateLegalMoves()) {
                if (!isZeroing(move)) { // Implement isZeroing to check if the move resets the
                                        // fifty-move counter
                    push(move);
                    bool result = isFiftyMoves();
                    pop();
                    if (result)
                        return true;
                }
            }
        }

        return false;
    }

    bool canClaimThreefoldRepetition() {
        std::unordered_map<size_t, int> transpositions;
        size_t transpositionKey = _transpositionKey();
        transpositions[transpositionKey]++;

        // Count positions
        std::vector<Move> switchyard;
        while (!move_stack.empty()) {
            Move move = pop();
            switchyard.push_back(move);

            if (isIrreversible(move))
                break;

            transpositions[_transpositionKey()]++;
        }

        while (!switchyard.empty()) {
            push(switchyard.back());
            switchyard.pop_back();
        }

        // Threefold repetition occurred
        if (transpositions[transpositionKey] >= 3) {
            return true;
        }

        // The next legal move is a threefold repetition
        for (const Move &move : generateLegalMoves()) {
            push(move);
            bool repetition = transpositions[_transpositionKey()] >= 2;
            pop();
            if (repetition)
                return true;
        }

        return false;
    }

    bool isRepetition(int count = 3) {
        // Fast check based on occupancy only
        int maybeRepetitions = 1;
        for (auto it = _stack.rbegin(); it != _stack.rend() && maybeRepetitions < count; ++it) {
            if ((*it)->occupied == occupied) {
                maybeRepetitions++;
            }
        }
        if (maybeRepetitions < count) {
            return false;
        }

        // Check full replay
        std::vector<Move> switchyard;
        size_t transpositionKey = _transpositionKey();

        try {
            while (true) {
                if (count <= 1)
                    return true;
                if (move_stack.size() < static_cast<size_t>(count - 1))
                    break;

                Move move = pop();
                switchyard.push_back(move);

                if (isIrreversible(move))
                    break;

                if (_transpositionKey() == transpositionKey) {
                    count--;
                }
            }
        } catch (...) {
            // Ensure we reapply all moves in switchyard even if an exception occurs
        }

        for (auto it = switchyard.rbegin(); it != switchyard.rend(); ++it) {
            push(*it);
        }

        return false;
    }

    // Updates the position with the given move and puts it onto the move stack
    void push(const Move &move) {
        // Capture the current board state before making the move
        _BoardState board_state(*this);
        _stack.push_back(board_state);

        // Reset en passant square, handle turn and fullmove counter
        auto ep_square = this->ep_square;
        this->ep_square = std::nullopt;
        halfmove_clock++;
        if (turn == BLACK) {
            fullmove_number++;
        }

        // On a null move, simply swap turns and return
        if (!move) {
            turn = !turn;
            return;
        }

        // Specific move handling logic goes here
        // This includes updating pieces' positions, handling captures, promotions, en passant,
        // castling, etc.
        // ...

        // Swap turn at the end of the move
        turn = !turn;

        // Add the move to the move stack
        move_stack.push_back(move);
    }

    // Restores the previous position and returns the last move from the stack
    Move pop() {
        if (move_stack.empty()) {
            throw std::out_of_range("Move stack is empty.");
        }

        Move last_move = move_stack.back();
        move_stack.pop_back();

        // Restore the board state to its previous configuration
        if (!_stack.empty()) {
            _stack.back().restore(*this);
            _stack.pop_back();
        }

        return last_move;
    }

    // Gets the last move from the move stack
    Move peek() const {
        if (move_stack.empty()) {
            throw std::out_of_range("Move stack is empty.");
        }

        return move_stack.back();
    }

    Move findMove(int from_square, int to_square,
                  std::optional<PieceType> promotion = std::nullopt) {
        if (!promotion.has_value() && (pawns & BB_SQUARES[from_square]) &&
            (BB_SQUARES[to_square] & BB_BACKRANKS)) {
            promotion = PieceType::QUEEN;
        }

        // Assuming a method to create a Move object and check legality exists
        Move move(from_square, to_square, promotion);
        if (!isLegal(move)) {
            throw IllegalMoveError("No matching legal move found in the current position.");
        }

        return move;
    }
    bool hasPseudoLegalEnPassant() {
        if (!ep_square.has_value())
            return false;

        auto epMoves = generatePseudoLegalEP();
        return !epMoves.empty();
    }
    bool hasLegalEnPassant() {
        if (!ep_square.has_value())
            return false;

        auto epMoves = generateLegalEP(); // Assuming generateLegalEP method exists
        return !epMoves.empty();
    }
    void setPieceMap(const std::unordered_map<Square, Piece> &pieces) {
        BaseBoard::setPieceMap(pieces); // Assuming BaseBoard class has a similar method
        clearStack();
    }

    std::string san(const Move &move) { return _algebraic(move, false); }

    std::string lan(const Move &move) { return _algebraic(move, true); }

    std::string sanAndPush(const Move &move) {
        std::string notation = _algebraicAndPush(move, false);
        // push(move) is assumed to be called within _algebraicAndPush
        return notation;
    }

    std::string _algebraic(const Move &move, bool longNotation = false) {
        std::string notation = _algebraicAndPush(move, longNotation);
        pop(); // Assume pop() undoes the last move made by push within _algebraicAndPush
        return notation;
    }

    std::string _algebraicAndPush(const Move &move, bool longNotation = false) {
        push(move); // Assume push() makes the move on the board

        std::string notation = _algebraicWithoutSuffix(move, longNotation);
        bool check = isCheck(); // Assume isCheck() checks if the current player is in check
        bool checkmate = check && (isCheckmate() || isVariantLoss() ||
                                   isVariantWin()); // Assume these methods are defined

        if (checkmate) {
            notation += '#';
        } else if (check) {
            notation += '+';
        }

        return notation;
    }

    std::string algebraicWithoutSuffix(const Move &move, bool longNotation = false) {
        // Null move
        if (!move) {
            return "--";
        }

        // Drops are not a standard part of classical chess, so omitted here.
        // Castling
        if (isCastling(move)) {
            if (square_file(move.toSquare()) < square_file(move.fromSquare())) {
                return "O-O-O";
            } else {
                return "O-O";
            }
        }

        std::optional<PieceType> piece_type = pieceTypeAt(move.fromSquare());
        assert(piece_type.has_value()); // Ensure the move is not null
        bool capture = isCapture(move);

        std::string san;
        if (piece_type.value() == PieceType::PAWN) {
            san = "";
        } else {
            san = piece_symbol(piece_type.value()).upper();
        }

        if (longNotation) {
            san += SQUARE_NAMES[move.fromSquare()];
        } else if (piece_type.value() != PAWN) {
            // Disambiguation
            Bitboard others = 0;
            Bitboard from_mask = piecesMask(piece_type.value(), turn);
            from_mask &= ~BB_SQUARES[move.fromSquare()];
            Bitboard to_mask = BB_SQUARES[move.toSquare()];
            for (const Move &candidate : generateLegalMoves(from_mask, to_mask)) {
                if (candidate != move) {
                    others |= BB_SQUARES[candidate.fromSquare()];
                }
            }

            bool row = false, column = false;
            if (others & BB_RANKS[square_rank(move.fromSquare())]) {
                column = true;
            }
            if (others & BB_FILES[square_file(move.fromSquare())]) {
                row = true;
            } else {
                column = true; // Always disambiguate by file unless it's unnecessary
            }

            if (column) {
                san += FILE_NAMES[square_file(move.fromSquare())];
            }
            if (row) {
                san += RANK_NAMES[square_rank(move.fromSquare())];
            }
        } else if (capture) {
            san += FILE_NAMES[square_file(move.fromSquare())];
        }

        // Captures
        if (capture) {
            san += 'x';
        } else if (longNotation) {
            san += '-';
        }

        // Destination
        san += SQUARE_NAMES[move.toSquare()];

        // Promotion
        if (move.promotion()) {
            san += '=' + piece_symbol(move.promotion().value()).upper();
        }

        return san;
    }

    bool isEnPassant(const Move &move) {
        return ep_square.has_value() && move.to_square == ep_square.value() &&
               (pawns & BB_SQUARES[move.from_square]) &&
               std::abs(move.to_square - move.from_square) in{7, 9} &&
               !(occupied & BB_SQUARES[move.to_square]);
    }

    bool isCapture(const Move &move) {
        Bitboard touched = BB_SQUARES[move.from_square] ^ BB_SQUARES[move.to_square];
        return (touched & occupied_co[!turn]) || isEnPassant(move);
    }

    bool isZeroing(const Move &move) {
        Bitboard touched = BB_SQUARES[move.from_square] ^ BB_SQUARES[move.to_square];
        return (touched & pawns) || (touched & occupied_co[!turn]) ||
               (move.promotion.has_value() && move.promotion == PAWN);
    }

    bool reducesCastlingRights(const Move &move) {
        Bitboard cr = cleanCastlingRights();
        Bitboard touched = BB_SQUARES[move.from_square] ^ BB_SQUARES[move.to_square];
        return (touched & cr) ||
               (cr & BB_RANK_1 && touched & kings & occupied_co[WHITE] & ~promoted) ||
               (cr & BB_RANK_8 && touched & kings & occupied_co[BLACK] & ~promoted);
    }

    bool isIrreversible(const Move &move) {
        return isZeroing(move) || reducesCastlingRights(move) || hasLegalEnPassant();
    }

    bool isCastling(const Move &move) {
        if (kings & BB_SQUARES[move.from_square]) {
            int diff = square_file(move.from_square) - square_file(move.to_square);
            return std::abs(diff) > 1 || (rooks & occupied_co[turn] & BB_SQUARES[move.to_square]);
        }
        return false;
    }

    bool isKingsideCastling(const Move &move) {
        return isCastling(move) && square_file(move.to_square) > square_file(move.from_square);
    }

    bool isQueensideCastling(const Move &move) {
        return isCastling(move) && square_file(move.to_square) < square_file(move.from_square);
    }

    Bitboard cleanCastlingRights() {
        if (!_stack.empty()) {
            return castling_rights;
        }

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

    bool hasCastlingRights(Color color) {
        Bitboard backrank = (color == WHITE) ? BB_RANK_1 : BB_RANK_8;
        return bool(cleanCastlingRights() & backrank);
    }

    bool hasKingsideCastlingRights(Color color) {
        Bitboard backrank = (color == WHITE) ? BB_RANK_1 : BB_RANK_8;
        Bitboard kingMask = kings & occupied_co[color] & backrank & ~promoted;
        if (kingMask == BB_EMPTY) {
            return false;
        }

        Bitboard castlingRights = cleanCastlingRights() & backrank;
        while (castlingRights) {
            Bitboard rook = castlingRights &
                            -castlingRights; // Isolate the least-significant bit (rightmost rook)

            if (rook > kingMask) { // Kingside rook is to the right of the king
                return true;
            }

            castlingRights &= castlingRights - 1; // Remove the least-significant bit
        }

        return false;
    }

    bool hasQueensideCastlingRights(Color color) {
        Bitboard backrank = (color == WHITE) ? BB_RANK_1 : BB_RANK_8;
        Bitboard kingMask = kings & occupied_co[color] & backrank & ~promoted;
        if (kingMask == BB_EMPTY) {
            return false;
        }

        Bitboard castlingRights = cleanCastlingRights() & backrank;
        while (castlingRights) {
            Bitboard rook = castlingRights &
                            -castlingRights; // Isolate the least-significant bit (leftmost rook)

            if (rook < kingMask) { // Queenside rook is to the left of the king
                return true;
            }

            castlingRights &= castlingRights - 1; // Remove the least-significant bit
        }

        return false;
    }

    Status status() {
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
        if (pawns & BB_BACKRANKS) {
            errors |= Status::PAWNS_ON_BACKRANK;
        }

        // Castling rights
        if (castling_rights != clean_castling_rights()) {
            errors |= Status::BAD_CASTLING_RIGHTS;
        }

        // En passant
        auto valid_ep_square = _valid_ep_square();
        if (ep_square != valid_ep_square) {
            errors |= Status::INVALID_EP_SQUARE;
        }

        // Check conditions
        if (was_into_check()) {
            errors |= Status::OPPOSITE_CHECK;
        }

        Bitboard checkers = checkers_mask();
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
                (msb(checkers) != pushed_to && _attacked_for_king(our_kings, occupied_before))) {
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

    std::optional<Square> validEPSquare() {
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

    bool isValid() {
        // Assuming STATUS_VALID and a status() method implementation
        return status() == STATUS_VALID;
    }

    bool epSkewered(Square king, Square capturer) {
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

    Bitboard sliderBlockers(Square king) {
        Bitboard rooks_and_queens = rooks | queens;
        Bitboard bishops_and_queens = bishops | queens;

        Bitboard snipers = ((BB_RANK_ATTACKS[king][0] & rooks_and_queens) |
                            (BB_FILE_ATTACKS[king][0] & rooks_and_queens) |
                            (BB_DIAG_ATTACKS[king][0] & bishops_and_queens)) &
                           occupied_co[!turn];

        Bitboard blockers = BB_EMPTY;
        for (Square sniper : scan_reversed(snipers)) {
            Bitboard b = between(king, sniper) & occupied;
            if (b && popcount(b) == 1) {
                blockers |= b;
            }
        }

        return blockers & occupied_co[turn];
    }

    bool isSafe(int king, Bitboard blockers, const Move &move) {
        if (move.fromSquare == king) {
            if (isCastling(move)) {
                return true;
            } else {
                return !isAttackedBy(!turn, move.toSquare);
            }
        } else if (isEnPassant(move)) {
            return (pinMask(turn, move.fromSquare) & BB_SQUARES[move.toSquare]) &&
                   !_epSkewered(king, move.fromSquare);
        } else {
            return !(blockers & BB_SQUARES[move.fromSquare]) ||
                   (ray(move.fromSquare, move.toSquare) & BB_SQUARES[king]);
        }
    }

    std::vector<Move> generateEvasions(int king, Bitboard checkers, Bitboard from_mask = BB_ALL,
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
            auto directEvasions = generatePseudoLegalMoves(~kings & from_mask, target & to_mask);
            evasions.insert(evasions.end(), directEvasions.begin(), directEvasions.end());

            // Capture the checking pawn en passant (avoiding duplicates).
            if (ep_square.has_value() && !(BB_SQUARES[ep_square.value()] & target)) {
                int last_double = ep_square.value() + (turn == WHITE ? -8 : 8);
                if (last_double == checker) {
                    auto epEvasions = generatePseudoLegalEP(from_mask, to_mask);
                    evasions.insert(evasions.end(), epEvasions.begin(), epEvasions.end());
                }
            }
        }

        return evasions;
    }

    std::vector<Move> generateLegalMoves(Bitboard from_mask = BB_ALL, Bitboard to_mask = BB_ALL) {
        if (isVariantEnd()) {
            return {};
        }

        std::vector<Move> legalMoves;
        Bitboard king_mask = kings & occupied_co[turn];
        if (king_mask) {
            int king = msb(king_mask); // Assuming msb() finds the most significant bit.
            Bitboard blockers = sliderBlockers(king); // Assuming sliderBlockers() is implemented.
            Bitboard checkers =
                attackersMask(!turn, king); // Assuming attackersMask() is implemented.
            if (checkers) {
                auto evasions =
                    generateEvasions(king, checkers, from_mask,
                                     to_mask); // Assuming generateEvasions() is implemented.
                std::copy_if(evasions.begin(), evasions.end(), std::back_inserter(legalMoves),
                             [this, king, blockers](const Move &move) {
                                 return isSafe(king, blockers,
                                               move); // Assuming isSafe() is implemented.
                             });
            } else {
                auto pseudoLegalMoves = generatePseudoLegalMoves(from_mask, to_mask);
                std::copy_if(pseudoLegalMoves.begin(), pseudoLegalMoves.end(),
                             std::back_inserter(legalMoves),
                             [this, king, blockers](const Move &move) {
                                 return isSafe(king, blockers, move);
                             });
            }
        } else {
            auto pseudoLegalMoves = generatePseudoLegalMoves(from_mask, to_mask);
            std::copy(pseudoLegalMoves.begin(), pseudoLegalMoves.end(),
                      std::back_inserter(legalMoves));
        }
        return legalMoves;
    }

    std::vector<Move> generateLegalEP(Bitboard from_mask = BB_ALL, Bitboard to_mask = BB_ALL) {
        std::vector<Move> legalEP;
        auto pseudoLegalEP = generatePseudoLegalEP(from_mask, to_mask);
        std::copy_if(pseudoLegalEP.begin(), pseudoLegalEP.end(), std::back_inserter(legalEP),
                     [this](const Move &move) { return !isIntoCheck(move); });
        return legalEP;
    }

    std::vector<Move> generateLegalCaptures(Bitboard from_mask = BB_ALL,
                                            Bitboard to_mask = BB_ALL) {
        std::vector<Move> legalMoves = generateLegalMoves(from_mask, to_mask & occupied_co[!turn]);
        std::vector<Move> legalEP = generateLegalEP(from_mask, to_mask);

        legalMoves.insert(legalMoves.end(), legalEP.begin(), legalEP.end());
        return legalMoves;
    }
    bool _attacked_for_king(Bitboard path, Bitboard occupied) {
        for (int sq = msb(path); path; sq = msb(path)) {
            if (_attackers_mask(!turn, sq, occupied)) {
                return true;
            }
            path &= path - 1; // Clear the scanned bit
        }
        return false;
    }

    std::vector<Move> generate_castling_moves(Bitboard from_mask = BB_ALL,
                                              Bitboard to_mask = BB_ALL) {
        std::vector<Move> moves;
        if (is_variant_end()) {
            return moves;
        }

        Bitboard backrank = (turn == WHITE) ? BB_RANK_1 : BB_RANK_8;
        Bitboard king = occupied_co[turn] & kings & ~promoted & backrank & from_mask;
        if (!king) {
            return moves;
        }

        Bitboard bb_c = BB_FILE_C & backrank;
        Bitboard bb_d = BB_FILE_D & backrank;
        Bitboard bb_f = BB_FILE_F & backrank;
        Bitboard bb_g = BB_FILE_G & backrank;

        Bitboard clean_rights = clean_castling_rights();
        for (int candidate = msb(clean_rights & backrank & to_mask); clean_rights;
             candidate = msb(clean_rights)) {
            Bitboard rook = BB_SQUARES[candidate];

            bool a_side = rook < king;
            Bitboard king_to = a_side ? bb_c : bb_g;
            Bitboard rook_to = a_side ? bb_d : bb_f;

            Bitboard king_path = between(msb(king), msb(king_to));
            Bitboard rook_path = between(candidate, msb(rook_to));

            if (!((occupied ^ king ^ rook) & (king_path | rook_path | king_to | rook_to)) &&
                !_attacked_for_king(king_path | king, occupied ^ king) &&
                !_attacked_for_king(king_to, occupied ^ king ^ rook ^ rook_to)) {
                moves.emplace_back(
                    candidate, msb(king_to)); // Assuming Move constructor accepts from/to squares
            }
            clean_rights &= clean_rights - 1; // Clear the scanned bit
        }
        return moves;
    }
    using TranspositionKey = std::tuple<Bitboard, Bitboard, Bitboard, Bitboard, Bitboard, Bitboard,
                                        Bitboard, Bitboard, Color, Bitboard, std::optional<Square>>;

    TranspositionKey _transposition_key() {
        return {pawns,
                knights,
                bishops,
                rooks,
                queens,
                kings,
                occupied_co[WHITE],
                occupied_co[BLACK],
                turn,
                clean_castling_rights(),
                has_legal_en_passant() ? ep_square : std::optional<Square>{}};
    }

    bool operator==(const Board &other) const {
        return halfmove_clock == other.halfmove_clock && fullmove_number == other.fullmove_number &&
               uci_variant == other.uci_variant &&
               _transposition_key() == other._transposition_key();
    }

    // Applies a transformation function to the board
    void applyTransform(const std::function<Bitboard(Bitboard)> &f) {
        BaseBoard::applyTransform(f); // Assuming BaseBoard has an applyTransform method
        clearStack();
        ep_square = (!ep_square.has_value())
                        ? std::nullopt
                        : std::optional<Square>(msb(f(BB_SQUARES[ep_square.value()])));
        castling_rights = f(castling_rights);
    }

    // Returns a transformed copy of the board
    Board transform(const std::function<Bitboard(Bitboard)> &f) {
        Board board = this->copy(false); // Copy without the stack
        board.applyTransform(f);
        return board;
    }

    // Applies a mirror transformation to the board
    void applyMirror() {
        BaseBoard::applyMirror();
        turn = !turn;
    }

    // Returns a mirrored copy of the board
    Board mirror() {
        Board board = this->copy();
        board.applyMirror();
        return board;
    }

    // Creates a copy of the board
    Board copy(bool copyStack = true) {
        Board board(std::o) if (!copyStack) {
            board.move_stack.clear();
            board._stack.clear();
        }
        else {
            // Optionally limit the stack copying depth
            // Not directly supported as in Python, but could be implemented if needed
        }
        return board;
    }
};

class LegalMoveGenerator {
private:
    Board *board; // Pointer to a Board object

public:
    // Constructor
    explicit LegalMoveGenerator(Board *board) : board(board) {}

    // Check if there are any legal moves
    bool hasLegalMoves() const {
        // Assuming Board::generateLegalMoves() returns a std::vector<Move>
        auto moves = board->generateLegalMoves();
        return !moves.empty();
    }

    // Count legal moves
    std::size_t count() const {
        auto moves = board->generateLegalMoves();
        return moves.size();
    }

    // Get legal moves
    std::vector<Move> getLegalMoves() const { return board->generateLegalMoves(); }

    // Check if a move is legal
    bool contains(const Move &move) const { return board->isLegal(move); }

    // Representation
    std::string repr() const {
        std::stringstream ss;
        ss << "<LegalMoveGenerator at " << this << " (";
        auto moves = getLegalMoves();
        for (auto it = moves.begin(); it != moves.end(); ++it) {
            if (it != moves.begin()) {
                ss << ", ";
            }
            ss << board->san(*it);
        }
        ss << ")>";
        return ss.str();
    }
};

class PseudoLegalMoveGenerator {
private:
    Board *board; // Pointer to a Board object

public:
    // Constructor
    explicit PseudoLegalMoveGenerator(Board *board) : board(board) {}

    // Check if there are any pseudo-legal moves
    bool hasPseudoLegalMoves() const {
        // Assuming Board::generatePseudoLegalMoves() returns a std::vector<Move>
        auto moves = board->generatePseudoLegalMoves();
        return !moves.empty();
    }

    // Count pseudo-legal moves
    std::size_t count() const {
        auto moves = board->generatePseudoLegalMoves();
        return moves.size();
    }

    // Get pseudo-legal moves
    std::vector<Move> getPseudoLegalMoves() const { return board->generatePseudoLegalMoves(); }

    // Check if a move is pseudo-legal
    bool contains(const Move &move) const { return board->isPseudoLegal(move); }

    // Representation
    std::string repr() const {
        std::stringstream ss;
        ss << "<PseudoLegalMoveGenerator at " << this << " (";
        auto moves = getPseudoLegalMoves();
        for (auto it = moves.begin(); it != moves.end(); ++it) {
            if (it != moves.begin()) {
                ss << ", ";
            }
            if (board->isLegal(*it)) {
                ss << board->san(*it); // Assuming Board::san() converts a Move to a standard
                                       // algebraic notation string
            } else {
                ss << board->uci(*it); // Assuming Board::uci() converts a Move to a universal chess
                                       // interface notation string
            }
        }
        ss << ")>";
        return ss.str();
    }
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
            if (square_rank(square) != 7) {
                result += " ";
            }
            if (square_file(square) == 7) {
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