#pragma once

#include "Engine/Move Generation/Bitboards/BitBoardUtility.h"
#include "Engine/Move Generation/Magics/Magic.h"
#include "GameState.h"
#include "Move.h"
#include "Piece.h"
#include "PieceList.h"
#include "Zobrist.h"
#include <array>
#include <memory>
#include <stack>
#include <string>
#include <vector>

namespace Chess {
// Represents the current state of the board during a game.
// The state includes things such as: positions of all pieces, side to move,
// castling rights, en-passant square, etc. Some extra information is included
// as well to help with evaluation and move generation.

// The initial state of the board can be set from a FEN string, and moves are
// subsequently made (or undone) using the MakeMove and UnmakeMove functions.

class Board {

public:
    static inline constexpr int WhiteIndex = 0;
    static inline constexpr int BlackIndex = 1;

    // Stores piece code for each square on the board

    std::array<int, 64> Square;
    // Square index of white and black king

    std::array<int, 2> KingSquare;
    // # Bitboards

    // Bitboard for each piece type and colour (white pawns, white knights, ... black pawns, etc.)
    std::array<unsigned long long, Piece::TotalTypesOfPieces> PieceBitboards;

    // Bitboards for all pieces of either colour (all white pieces, all black pieces)
    std::array<unsigned long long, 2> ColourBitboards;

    unsigned long long AllPiecesBitboard = 0;

    unsigned long long FriendlyOrthogonalSliders = 0;

    unsigned long long FriendlyDiagonalSliders = 0;

    unsigned long long EnemyOrthogonalSliders = 0;

    unsigned long long EnemyDiagonalSliders = 0;

    // Piece count excluding pawns and kings
    int TotalPieceCountWithoutPawnsAndKings = 0;

    // # Piece lists
    std::array<PieceList, 2> Rooks = {PieceList(10), PieceList(10)};
    std::array<PieceList, 2> Bishops = {PieceList(10), PieceList(10)};
    std::array<PieceList, 2> Queens = {PieceList(9), PieceList(9)};
    std::array<PieceList, 2> Knights = {PieceList(10), PieceList(10)};
    std::array<PieceList, 2> Pawns = {PieceList(8), PieceList(8)};

    // # Side to move info

    bool IsWhiteToMove = false;

    int MoveColour() const { return IsWhiteToMove ? Piece::White : Piece::Black; }

    int OpponentColour() const { return IsWhiteToMove ? Piece::Black : Piece::White; }

    int MoveColourIndex() const { return IsWhiteToMove ? WhiteIndex : BlackIndex; }

    int OpponentColourIndex() const { return IsWhiteToMove ? BlackIndex : WhiteIndex; }

    int FiftyMoveCounter() const { return CurrentGameState.fiftyMoveCounter; }

    unsigned long long ZobristKey() const { return CurrentGameState.zobristKey; }

    // List of (hashed) positions since last pawn move or capture (for detecting repetitions)
    std::stack<unsigned long long> RepetitionPositionHistory;

    // Total plies (half-moves) played in game
    int PlyCount = 0;

    GameState CurrentGameState;

    // # public stuff
    std::array<PieceList &, Piece::TotalTypesOfPieces> allPieceLists;
    std::stack<GameState> gameStateHistory;

    bool cachedInCheckValue = false;
    bool hasCachedInCheckValue = false;

    // Make a move on the board
    // The inSearch parameter controls whether this move should be recorded in the game history.
    // (for detecting three-fold repetition)

    void MakeMove(Move move, bool inSearch = false) {
        // Get info about move
        int startSquare = move.StartSquare();
        int targetSquare = move.TargetSquare();
        int moveFlag = move.MoveFlag();
        bool IsPromotion = move.IsPromotion();
        bool isEnPassant = moveFlag == Move::EnPassantCaptureFlag;

        int movedPiece = Square[startSquare];
        int movedPieceType = Piece::PieceType(movedPiece);
        int capturedPiece =
            isEnPassant ? Piece::MakePiece(Piece::Pawn, OpponentColour) : Square[targetSquare];
        int capturedPieceType = Piece::PieceType(capturedPiece);

        int prevCastleState = CurrentGameState.castlingRights;
        int prevEnPassantFile = CurrentGameState.enPassantFile;
        unsigned long long newZobristKey = CurrentGameState.zobristKey;
        int newCastlingRights = CurrentGameState.castlingRights;
        int newEnPassantFile = 0;

        // Update bitboard of moved piece (pawn promotion is a special case and is corrected later)
        MovePiece(movedPiece, startSquare, targetSquare);

        // Handle captures
        if (capturedPieceType != Piece::None) {
            int captureSquare = targetSquare;

            if (isEnPassant) {
                captureSquare = targetSquare + (IsWhiteToMove ? -8 : 8);
                Square[captureSquare] = Piece::None;
            }
            if (capturedPieceType != Piece::Pawn) {
                TotalPieceCountWithoutPawnsAndKings--;
            }

            // Remove captured piece from bitboards/piece list
            allPieceLists[capturedPiece].RemovePieceAtSquare(captureSquare);
            BitBoardUtility::ToggleSquare(PieceBitboards[capturedPiece], captureSquare);
            BitBoardUtility::ToggleSquare(ColourBitboards[OpponentColourIndex()], captureSquare);
            newZobristKey ^= Zobrist::piecesArray[capturedPiece][captureSquare];
        }

        // Handle king
        if (movedPieceType == Piece::King) {
            KingSquare[MoveColourIndex()] = targetSquare;
            newCastlingRights &= (IsWhiteToMove) ? 0b1100 : 0b0011;

            // Handle castling
            if (moveFlag == Move::CastleFlag) {
                int rookPiece = Piece::MakePiece(Piece::Rook, MoveColour);
                bool kingside = targetSquare == BoardHelper::g1 || targetSquare == BoardHelper::g8;
                int castlingRookFromIndex = (kingside) ? targetSquare + 1 : targetSquare - 2;
                int castlingRookToIndex = (kingside) ? targetSquare - 1 : targetSquare + 1;

                // Update rook position
                BitBoardUtility::ToggleSquares(PieceBitboards[rookPiece], castlingRookFromIndex,
                                               castlingRookToIndex);
                BitBoardUtility::ToggleSquares(ColourBitboards[MoveColourIndex()],
                                               castlingRookFromIndex, castlingRookToIndex);
                allPieceLists[rookPiece].MovePiece(castlingRookFromIndex, castlingRookToIndex);
                Square[castlingRookFromIndex] = Piece::None;
                Square[castlingRookToIndex] = Piece::Rook | MoveColour();

                newZobristKey ^= Zobrist::piecesArray[rookPiece][castlingRookFromIndex];
                newZobristKey ^= Zobrist::piecesArray[rookPiece][castlingRookToIndex];
            }
        }

        // Handle promotion
        if (IsPromotion) {
            TotalPieceCountWithoutPawnsAndKings++;

            int promotionPieceType = 0;
            switch (moveFlag) {
            case Move::PromoteToQueenFlag:
                promotionPieceType = Piece::Queen;
                break;
            case Move::PromoteToRookFlag:
                promotionPieceType = Piece::Rook;
                break;
            case Move::PromoteToKnightFlag:
                promotionPieceType = Piece::Knight;
                break;
            case Move::PromoteToBishopFlag:
                promotionPieceType = Piece::Bishop;
                break;
            default:
                promotionPieceType = 0;
                break;
            }

            int promotionPiece = Piece::MakePiece(promotionPieceType, MoveColour);

            // Remove pawn from promotion square and add promoted piece instead
            BitBoardUtility::ToggleSquare(PieceBitboards[movedPiece], targetSquare);
            BitBoardUtility::ToggleSquare(PieceBitboards[promotionPiece], targetSquare);
            allPieceLists[movedPiece].RemovePieceAtSquare(targetSquare);
            allPieceLists[promotionPiece].AddPieceAtSquare(targetSquare);
            Square[targetSquare] = promotionPiece;
        }

        // Pawn has moved two forwards, mark file with en-passant flag
        if (moveFlag == Move::PawnTwoUpFlag) {
            int file = BoardHelper::FileIndex(startSquare) + 1;
            newEnPassantFile = file;
            newZobristKey ^= Zobrist::enPassantFile[file];
        }

        // Update castling rights
        if (prevCastleState != 0) {
            // Any piece moving to/from rook square removes castling right for that side
            if (targetSquare == BoardHelper::h1 || startSquare == BoardHelper::h1) {
                newCastlingRights &= GameState::ClearWhiteKingsideMask;
            } else if (targetSquare == BoardHelper::a1 || startSquare == BoardHelper::a1) {
                newCastlingRights &= GameState::ClearWhiteQueensideMask;
            }
            if (targetSquare == BoardHelper::h8 || startSquare == BoardHelper::h8) {
                newCastlingRights &= GameState::ClearBlackKingsideMask;
            } else if (targetSquare == BoardHelper::a8 || startSquare == BoardHelper::a8) {
                newCastlingRights &= GameState::ClearBlackQueensideMask;
            }
        }

        // Update zobrist key with piece position and side to move
        newZobristKey ^= Zobrist::sideToMove;
        newZobristKey ^= Zobrist::piecesArray[movedPiece][startSquare];
        newZobristKey ^= Zobrist::piecesArray[Square[targetSquare]][targetSquare];
        newZobristKey ^= Zobrist::enPassantFile[prevEnPassantFile];

        if (newCastlingRights != prevCastleState) {
            newZobristKey ^=
                Zobrist::castlingRights[prevCastleState]; // remove old castling rights state
            newZobristKey ^=
                Zobrist::castlingRights[newCastlingRights]; // add castling rights state
        }

        // Change side to move
        IsWhiteToMove = !IsWhiteToMove;

        PlyCount++;
        int newFiftyMoveCounter = CurrentGameState.fiftyMoveCounter + 1;

        // Update extra bitboards
        AllPiecesBitboard = ColourBitboards[WhiteIndex] | ColourBitboards[BlackIndex];
        UpdateSliderBitboards();

        // Pawn moves and captures reset the fifty move counter and clear 3-fold repetition history
        if (movedPieceType == Piece::Pawn || capturedPieceType != Piece::None) {
            if (!inSearch) {
                while (RepetitionPositionHistory.size() > 0) {
                    RepetitionPositionHistory.pop();
                }
            }
            newFiftyMoveCounter = 0;
        }

        GameState newState(capturedPieceType, newEnPassantFile, newCastlingRights,
                           newFiftyMoveCounter, newZobristKey);
        gameStateHistory.push(newState);
        CurrentGameState = newState;
        hasCachedInCheckValue = false;

        if (!inSearch) {
            RepetitionPositionHistory.push(newState.zobristKey);
            AllGameMoves.push_back(move);
        }
    }

    // Undo a move previously made on the board

    void UnmakeMove(Move move, bool inSearch = false) {
        // Swap colour to move
        IsWhiteToMove = !IsWhiteToMove;

        bool undoingWhiteMove = IsWhiteToMove;

        // Get move info
        int movedFrom = move.StartSquare();
        int movedTo = move.TargetSquare();
        int moveFlag = move.MoveFlag();

        bool undoingEnPassant = moveFlag == Move::EnPassantCaptureFlag;
        bool undoingPromotion = move.IsPromotion();
        bool undoingCapture = CurrentGameState.capturedPieceType != Piece::None;

        int movedPiece =
            undoingPromotion ? Piece::MakePiece(Piece::Pawn, MoveColour()) : Square[movedTo];
        int movedPieceType = Piece::PieceType(movedPiece);
        int capturedPieceType = CurrentGameState.capturedPieceType;

        // If undoing promotion, then remove piece from promotion square and replace with pawn
        if (undoingPromotion) {
            int promotedPiece = Square[movedTo];
            int pawnPiece = Piece::MakePiece(Piece::Pawn, MoveColour());
            TotalPieceCountWithoutPawnsAndKings--;

            allPieceLists[promotedPiece].RemovePieceAtSquare(movedTo);
            allPieceLists[movedPiece].AddPieceAtSquare(movedTo);
            BitBoardUtility::ToggleSquare(PieceBitboards[promotedPiece], movedTo);
            BitBoardUtility::ToggleSquare(PieceBitboards[pawnPiece], movedTo);
        }

        MovePiece(movedPiece, movedTo, movedFrom);

        // Undo capture
        if (undoingCapture) {
            int captureSquare = movedTo;
            int capturedPiece = Piece::MakePiece(capturedPieceType, OpponentColour());

            if (undoingEnPassant) {
                captureSquare = movedTo + ((undoingWhiteMove) ? -8 : 8);
            }
            if (capturedPieceType != Piece::Pawn) {
                TotalPieceCountWithoutPawnsAndKings++;
            }

            // Add back captured piece
            BitBoardUtility::ToggleSquare(PieceBitboards[capturedPiece], captureSquare);
            BitBoardUtility::ToggleSquare(ColourBitboards[OpponentColourIndex()], captureSquare);
            allPieceLists[capturedPiece].AddPieceAtSquare(captureSquare);
            Square[captureSquare] = capturedPiece;
        }

        // Update king
        if (movedPieceType == Piece::King) {
            KingSquare[MoveColourIndex()] = movedFrom;

            // Undo castling
            if (moveFlag == Move::CastleFlag) {
                int rookPiece = Piece::MakePiece(Piece::Rook, MoveColour());
                bool kingside = movedTo == BoardHelper::g1 || movedTo == BoardHelper::g8;
                int rookSquareBeforeCastling = kingside ? movedTo + 1 : movedTo - 2;
                int rookSquareAfterCastling = kingside ? movedTo - 1 : movedTo + 1;

                // Undo castling by returning rook to original square
                BitBoardUtility::ToggleSquares(PieceBitboards[rookPiece], rookSquareAfterCastling,
                                               rookSquareBeforeCastling);
                BitBoardUtility::ToggleSquares(ColourBitboards[MoveColourIndex()],
                                               rookSquareAfterCastling, rookSquareBeforeCastling);
                Square[rookSquareAfterCastling] = Piece::None;
                Square[rookSquareBeforeCastling] = rookPiece;
                allPieceLists[rookPiece].MovePiece(rookSquareAfterCastling,
                                                   rookSquareBeforeCastling);
            }
        }

        AllPiecesBitboard = ColourBitboards[WhiteIndex] | ColourBitboards[BlackIndex];
        UpdateSliderBitboards();

        if (!inSearch && RepetitionPositionHistory.size() > 0) {
            RepetitionPositionHistory.pop();
        }
        if (!inSearch) {
            AllGameMoves.pop_back();
        }

        // Go back to previous state
        gameStateHistory.pop();
        CurrentGameState = gameStateHistory.top();
        PlyCount--;
        hasCachedInCheckValue = false;
    }

    // Switch side to play without making a move (NOTE: must not be in check when called)

    void MakeNullMove() {
        IsWhiteToMove = !IsWhiteToMove;

        PlyCount++;

        unsigned long long newZobristKey = CurrentGameState.zobristKey;
        newZobristKey ^= Zobrist::sideToMove;
        newZobristKey ^= Zobrist::enPassantFile[CurrentGameState.enPassantFile];

        GameState newState(Piece::None, 0, CurrentGameState.castlingRights,
                           CurrentGameState.fiftyMoveCounter + 1, newZobristKey);
        CurrentGameState = newState;
        gameStateHistory.push(CurrentGameState);
        UpdateSliderBitboards();
        hasCachedInCheckValue = true;
        cachedInCheckValue = false;
    }

    void UnmakeNullMove() {
        IsWhiteToMove = !IsWhiteToMove;
        PlyCount--;
        gameStateHistory.pop();
        CurrentGameState = gameStateHistory.top();
        UpdateSliderBitboards();
        hasCachedInCheckValue = true;
        cachedInCheckValue = false;
    }

    // Is current player in check?
    // Note: caches check value so calling multiple times does not require recalculating

    bool IsInCheck() {
        if (hasCachedInCheckValue) {
            return cachedInCheckValue;
        }
        cachedInCheckValue = CalculateInCheckState();
        hasCachedInCheckValue = true;

        return cachedInCheckValue;
    }

    // Calculate in check value
    // Call IsInCheck instead for automatic caching of value

    bool CalculateInCheckState() {
        int kingSquare = KingSquare[MoveColourIndex()];
        unsigned long long blockers = AllPiecesBitboard;

        if (EnemyOrthogonalSliders != 0) {
            unsigned long long rookAttacks = Magic::GetRookAttacks(kingSquare, blockers);
            if ((rookAttacks & EnemyOrthogonalSliders) != 0) {
                return true;
            }
        }
        if (EnemyDiagonalSliders != 0) {
            unsigned long long bishopAttacks = Magic::GetBishopAttacks(kingSquare, blockers);
            if ((bishopAttacks & EnemyDiagonalSliders) != 0) {
                return true;
            }
        }

        unsigned long long enemyKnights =
            PieceBitboards[Piece::MakePiece(Piece::Knight, OpponentColour())];
        if ((BitBoardUtility::KnightAttacks[kingSquare] & enemyKnights) != 0) {
            return true;
        }

        unsigned long long enemyPawns =
            PieceBitboards[Piece::MakePiece(Piece::Pawn, OpponentColour())];
        unsigned long long pawnAttackMask = IsWhiteToMove
                                                ? BitBoardUtility::WhitePawnAttacks[kingSquare]
                                                : BitBoardUtility::BlackPawnAttacks[kingSquare];
        if ((pawnAttackMask & enemyPawns) != 0) {
            return true;
        }

        return false;
    }

    void LoadPosition(FenUtility::PositionInfo posInfo) {
        StartPositionInfo = posInfo;
        Initialize();

        // Load pieces into board array and piece lists
        for (int squareIndex = 0; squareIndex < 64; squareIndex++) {
            int piece = posInfo.squares[squareIndex];
            int pieceType = Piece::PieceType(piece);
            int colourIndex = Piece::IsWhite(piece) ? WhiteIndex : BlackIndex;
            Square[squareIndex] = piece;

            if (piece != Piece::None) {
                BitBoardUtility::SetSquare(PieceBitboards[piece], squareIndex);
                BitBoardUtility::SetSquare(ColourBitboards[colourIndex], squareIndex);

                if (pieceType == Piece::King) {
                    KingSquare[colourIndex] = squareIndex;
                } else {
                    allPieceLists[piece].AddPieceAtSquare(squareIndex);
                }
                TotalPieceCountWithoutPawnsAndKings +=
                    (pieceType == Piece::Pawn || pieceType == Piece::King) ? 0 : 1;
            }
        }

        // Side to move
        IsWhiteToMove = posInfo.whiteToMove;

        // Set extra bitboards
        AllPiecesBitboard = ColourBitboards[WhiteIndex] | ColourBitboards[BlackIndex];
        UpdateSliderBitboards();

        // Create game state
        int whiteCastle = ((posInfo.whiteCastleKingside) ? 1 << 0 : 0) |
                          ((posInfo.whiteCastleQueenside) ? 1 << 1 : 0);
        int blackCastle = ((posInfo.blackCastleKingside) ? 1 << 2 : 0) |
                          ((posInfo.blackCastleQueenside) ? 1 << 3 : 0);
        int castlingRights = whiteCastle | blackCastle;

        PlyCount = (posInfo.moveCount - 1) * 2 + (IsWhiteToMove ? 0 : 1);

        // Set game state (note: calculating zobrist key relies on current game state)
        CurrentGameState =
            GameState(Piece::None, posInfo.epFile, castlingRights, posInfo.fiftyMovePlyCount, 0);
        unsigned long long zobristKey = Zobrist::CalculateZobristKey(*this);
        CurrentGameState = GameState(Piece::None, posInfo.epFile, castlingRights,
                                     posInfo.fiftyMovePlyCount, zobristKey);

        RepetitionPositionHistory.push(zobristKey);

        gameStateHistory.push(CurrentGameState);
    }

    std::string ToString() { return BoardHelper::CreateDiagram(*this, IsWhiteToMove); }

    static Board CreateBoard(std::string fen = FenUtility::StartPositionFEN) {
        Board board;
        board.LoadPosition(fen);
        return board;
    }

    static Board CreateBoard(const Board &source) {
        Board board;
        board.LoadPosition(source.StartPositionInfo);

        for (int i = 0; i < source.AllGameMoves.size(); i++) {
            board.MakeMove(source.AllGameMoves[i]);
        }
        return board;
    }

    // Update piece lists / bitboards based on given move info.
    // Note that this does not account for the following things, which must be handled separately:
    // 1. Removal of a captured piece
    // 2. Movement of rook when castling
    // 3. Removal of pawn from 1st/8th rank during pawn promotion
    // 4. Addition of promoted piece during pawn promotion
    void MovePiece(int piece, int startSquare, int targetSquare) {
        BitBoardUtility::ToggleSquares(PieceBitboards[piece], startSquare, targetSquare);
        BitBoardUtility::ToggleSquares(ColourBitboards[MoveColourIndex()], startSquare,
                                       targetSquare);

        allPieceLists[piece].MovePiece(startSquare, targetSquare);
        Square[startSquare] = Piece::None;
        Square[targetSquare] = piece;
    }

    void UpdateSliderBitboards() {
        int friendlyRook = Piece::MakePiece(Piece::Rook, MoveColour());
        int friendlyQueen = Piece::MakePiece(Piece::Queen, MoveColour());
        int friendlyBishop = Piece::MakePiece(Piece::Bishop, MoveColour());
        FriendlyOrthogonalSliders = PieceBitboards[friendlyRook] | PieceBitboards[friendlyQueen];
        FriendlyDiagonalSliders = PieceBitboards[friendlyBishop] | PieceBitboards[friendlyQueen];

        int enemyRook = Piece::MakePiece(Piece::Rook, OpponentColour());
        int enemyQueen = Piece::MakePiece(Piece::Queen, OpponentColour());
        int enemyBishop = Piece::MakePiece(Piece::Bishop, OpponentColour());
        EnemyOrthogonalSliders = PieceBitboards[enemyRook] | PieceBitboards[enemyQueen];
        EnemyDiagonalSliders = PieceBitboards[enemyBishop] | PieceBitboards[enemyQueen];
    }

    void Initialize() {
        AllGameMoves = List<Move>();
        KingSquare = {0, 0};
        Square.fill(Piece::None);

        RepetitionPositionHistory = Stack<unsigned long long>(capacity : 64);
        gameStateHistory = Stack<GameState>(capacity : 64);

        CurrentGameState = GameState();
        PlyCount = 0;

        allPieceLists = PieceList[Piece::TotalTypesOfPieces];
        allPieceLists[Piece::WhitePawn] = Pawns[WhiteIndex];
        allPieceLists[Piece::WhiteKnight] = Knights[WhiteIndex];
        allPieceLists[Piece::WhiteBishop] = Bishops[WhiteIndex];
        allPieceLists[Piece::WhiteRook] = Rooks[WhiteIndex];
        allPieceLists[Piece::WhiteQueen] = Queens[WhiteIndex];
        allPieceLists[Piece::WhiteKing] = PieceList(1);

        allPieceLists[Piece::BlackPawn] = Pawns[BlackIndex];
        allPieceLists[Piece::BlackKnight] = Knights[BlackIndex];
        allPieceLists[Piece::BlackBishop] = Bishops[BlackIndex];
        allPieceLists[Piece::BlackRook] = Rooks[BlackIndex];
        allPieceLists[Piece::BlackQueen] = Queens[BlackIndex];
        allPieceLists[Piece::BlackKing] = PieceList(1);

        TotalPieceCountWithoutPawnsAndKings = 0;

        // Initialize bitboards
        PieceBitboards = unsigned long long[Piece::TotalTypesOfPieces];
        ColourBitboards = unsigned long long[2];
        AllPiecesBitboard = 0;
    }
};
} // namespace Chess