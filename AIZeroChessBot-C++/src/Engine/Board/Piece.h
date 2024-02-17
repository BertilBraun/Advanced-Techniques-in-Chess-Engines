#pragma once

#include <array>
#include <cctype>
#include <vector>

namespace Chess {
// Contains definitions for each piece type (represented as integers),
// as well as various helper functions for dealing with pieces.
class Piece final {
    // Piece Types
public:
    static constexpr int None = 0;
    static constexpr int Pawn = 1;
    static constexpr int Knight = 2;
    static constexpr int Bishop = 3;
    static constexpr int Rook = 4;
    static constexpr int Queen = 5;
    static constexpr int King = 6;

    // Piece Colours
    static constexpr int White = 0;
    static constexpr int Black = 8;

    // Pieces
    static constexpr int WhitePawn = Pawn | White;     // 1
    static constexpr int WhiteKnight = Knight | White; // 2
    static constexpr int WhiteBishop = Bishop | White; // 3
    static constexpr int WhiteRook = Rook | White;     // 4
    static constexpr int WhiteQueen = Queen | White;   // 5
    static constexpr int WhiteKing = King | White;     // 6

    static constexpr int BlackPawn = Pawn | Black;     // 9
    static constexpr int BlackKnight = Knight | Black; // 10
    static constexpr int BlackBishop = Bishop | Black; // 11
    static constexpr int BlackRook = Rook | Black;     // 12
    static constexpr int BlackQueen = Queen | Black;   // 13
    static constexpr int BlackKing = King | Black;     // 14

    static constexpr int MaxPieceIndex = BlackKing;
    static constexpr int TotalTypesOfPieces = BlackKing + 1;

    static constexpr std::array<int, TotalTypesOfPieces> PieceIndices = {
        WhitePawn, WhiteKnight, WhiteBishop, WhiteRook, WhiteQueen, WhiteKing,
        BlackPawn, BlackKnight, BlackBishop, BlackRook, BlackQueen, BlackKing};

    // Bit Masks
    static constexpr int typeMask = 0b0111;
    static constexpr int colourMask = 0b1000;

    static constexpr int MakePiece(int pieceType, int pieceColour) { return pieceType | pieceColour; }

    static constexpr int MakePiece(int pieceType, bool pieceIsWhite) {
        return MakePiece(pieceType, pieceIsWhite ? White : Black);
    }

    // Returns true if given piece matches the given colour. If piece is of type 'none', result will always be false.
    static constexpr bool IsColour(int piece, int colour) { return (piece & colourMask) == colour && piece != 0; }

    static constexpr bool IsWhite(int piece) { return IsColour(piece, White); }

    static constexpr int PieceColour(int piece) { return piece & colourMask; }

    static constexpr int PieceType(int piece) { return piece & typeMask; }

    // Rook or Queen
    static constexpr bool IsOrthogonalSlider(int piece) {
        int pieceType = PieceType(piece);
        return pieceType == Queen || pieceType == Rook;
    }

    // Bishop or Queen
    static constexpr bool IsDiagonalSlider(int piece) {
        int pieceType = PieceType(piece);
        return pieceType == Queen || pieceType == Bishop;
    }

    // Bishop, Rook, or Queen
    static constexpr bool IsSlidingPiece(int piece) {
        int pieceType = PieceType(piece);
        return pieceType == Queen || pieceType == Bishop || pieceType == Rook;
    }

    static constexpr char GetSymbol(int piece) {
        char symbol;
        switch (PieceType(piece)) {
        case Rook:
            symbol = 'R';
            break;
        case Knight:
            symbol = 'N';
            break;
        case Bishop:
            symbol = 'B';
            break;
        case Queen:
            symbol = 'Q';
            break;
        case King:
            symbol = 'K';
            break;
        case Pawn:
            symbol = 'P';
            break;
        default:
            symbol = ' ';
            break;
        }
        return IsWhite(piece) ? symbol : std::tolower(symbol);
    }

    static constexpr int GetPieceTypeFromSymbol(char symbol) {
        switch (std::toupper(symbol)) {
        case 'R':
            return Rook;
        case 'N':
            return Knight;
        case 'B':
            return Bishop;
        case 'Q':
            return Queen;
        case 'K':
            return King;
        case 'P':
            return Pawn;
        default:
            return None;
        }
    }
};
} // namespace Chess
