#pragma once

#include "Bitboards/BitBoardUtility.h"
#include "Bitboards/Bits.h"
#include "Engine/Board/Board.h"
#include "Engine/Board/Move.h"
#include "Engine/Board/Piece.h"
#include "Engine/Board/PieceList.h"
#include "Engine/Helpers/BoardHelper.h"
#include "Engine/Move Generation/Magics/Magic.h"
#include "Engine/Move Generation/PrecomputedMoveData.h"
#include <limits>
#include <memory>
#include <vector>

namespace Chess {
class MoveGenerator {
public:
    static constexpr int MaxMoves = 218;
    enum class PromotionMode { All, QueenOnly, QueenAndKnight };

    PromotionMode promotionsToGenerate = PromotionMode::All;

    // ---- Instance variables ----
    bool isWhiteToMove = false;
    int friendlyColour = 0;
    int opponentColour = 0;
    int friendlyKingSquare = 0;
    int friendlyIndex = 0;
    int enemyIndex = 0;

    bool inCheck = false;
    bool inDoubleCheck = false;

    // If in check, this bitboard contains squares in line from checking piece up to king
    // If not in check, all bits are set to 1
    unsigned long long checkRayBitmask = 0;

    unsigned long long pinRays = 0;
    unsigned long long notPinRays = 0;
    unsigned long long opponentAttackMapNoPawns = 0;
    unsigned long long opponentAttackMap = 0;
    unsigned long long opponentPawnAttackMap = 0;
    unsigned long long opponentSlidingAttackMap = 0;

    bool generateQuietMoves = false;
    Board const *board = nullptr;
    int currMoveIndex = 0;

    unsigned long long enemyPieces = 0;
    unsigned long long friendlyPieces = 0;
    unsigned long long allPieces = 0;
    unsigned long long emptySquares = 0;
    unsigned long long emptyOrEnemySquares = 0;
    // If only captures should be generated, this will have 1s only in positions of enemy pieces.
    // Otherwise it will have 1s everywhere.
    unsigned long long moveTypeMask = 0;

    MoveGenerator() {}

    std::vector<Move> GenerateMoves(const Board &board, bool capturesOnly = false) {
        std::vector<Move> moves = std::vector<Move>(MaxMoves);
        GenerateMoves(board, moves, capturesOnly);
        return moves;
    }

    // Generates list of legal moves in current position.
    // Quiet moves (non captures) can optionally be excluded. This is used in quiescence search.
    int GenerateMoves(const Board &board, std::vector<Move> &moves, bool capturesOnly = false) {
        this->board = &board;
        generateQuietMoves = !capturesOnly;

        Init();

        GenerateKingMoves(moves);

        // Only king moves are valid in a double check position, so can return early.
        if (!inDoubleCheck) {
            GenerateSlidingMoves(moves);
            GenerateKnightMoves(moves);
            GeneratePawnMoves(moves);
        }

        moves.resize(currMoveIndex);
        return moves.size();
    }

    // Note, this will only return correct value after GenerateMoves() has been called in the
    // current position
    bool InCheck() const { return inCheck; }

    void Init() {
        // Reset state
        currMoveIndex = 0;
        inCheck = false;
        inDoubleCheck = false;
        checkRayBitmask = 0;
        pinRays = 0;

        // Store some info for convenience
        isWhiteToMove = board->MoveColour() == Piece::White;
        friendlyColour = board->MoveColour();
        opponentColour = board->OpponentColour();
        friendlyKingSquare = board->KingSquare[board->MoveColourIndex()];
        friendlyIndex = board->MoveColourIndex();
        enemyIndex = 1 - friendlyIndex;

        // Store some bitboards for convenience
        enemyPieces = board->ColourBitboards[enemyIndex];
        friendlyPieces = board->ColourBitboards[friendlyIndex];
        allPieces = board->AllPiecesBitboard;
        emptySquares = ~allPieces;
        emptyOrEnemySquares = emptySquares | enemyPieces;
        moveTypeMask =
            generateQuietMoves ? std::numeric_limits<unsigned long long>::max() : enemyPieces;

        CalculateAttackData();
    }

    void GenerateKingMoves(std::vector<Move> &moves) {
        unsigned long long legalMask = ~(opponentAttackMap | friendlyPieces);
        unsigned long long kingMoves =
            BitBoardUtility::KingMoves[friendlyKingSquare] & legalMask & moveTypeMask;
        while (kingMoves != 0) {
            int targetSquare = BitBoardUtility::PopLSB(kingMoves);
            moves[currMoveIndex++] = Move(friendlyKingSquare, targetSquare);
        }

        // Castling
        if (!inCheck && generateQuietMoves) {
            unsigned long long castleBlockers = opponentAttackMap | board->AllPiecesBitboard;
            if (board->CurrentGameState.HasKingsideCastleRight(board->IsWhiteToMove)) {
                unsigned long long castleMask =
                    board->IsWhiteToMove ? Bits::WhiteKingsideMask : Bits::BlackKingsideMask;
                if ((castleMask & castleBlockers) == 0) {
                    int targetSquare = board->IsWhiteToMove ? BoardHelper::g1 : BoardHelper::g8;
                    moves[currMoveIndex++] =
                        Move(friendlyKingSquare, targetSquare, Move::CastleFlag);
                }
            }
            if (board->CurrentGameState.HasQueensideCastleRight(board->IsWhiteToMove)) {
                unsigned long long castleMask =
                    board->IsWhiteToMove ? Bits::WhiteQueensideMask2 : Bits::BlackQueensideMask2;
                unsigned long long castleBlockMask =
                    board->IsWhiteToMove ? Bits::WhiteQueensideMask : Bits::BlackQueensideMask;
                if ((castleMask & castleBlockers) == 0 &&
                    (castleBlockMask & board->AllPiecesBitboard) == 0) {
                    int targetSquare = board->IsWhiteToMove ? BoardHelper::c1 : BoardHelper::c8;
                    moves[currMoveIndex++] =
                        Move(friendlyKingSquare, targetSquare, Move::CastleFlag);
                }
            }
        }
    }

    void GenerateSlidingMoves(std::vector<Move> &moves) {
        // Limit movement to empty or enemy squares, and must block check if king is in check.
        unsigned long long moveMask = emptyOrEnemySquares & checkRayBitmask & moveTypeMask;

        unsigned long long othogonalSliders = board->FriendlyOrthogonalSliders;
        unsigned long long diagonalSliders = board->FriendlyDiagonalSliders;

        // Pinned pieces cannot move if king is in check
        if (inCheck) {
            othogonalSliders &= ~pinRays;
            diagonalSliders &= ~pinRays;
        }

        // Ortho
        while (othogonalSliders != 0) {
            int startSquare = BitBoardUtility::PopLSB(othogonalSliders);
            unsigned long long moveSquares =
                Magic::GetRookAttacks(startSquare, allPieces) & moveMask;

            // If piece is pinned, it can only move along the pin ray
            if (IsPinned(startSquare)) {
                moveSquares &= PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare];
            }

            while (moveSquares != 0) {
                int targetSquare = BitBoardUtility::PopLSB(moveSquares);
                moves[currMoveIndex++] = Move(startSquare, targetSquare);
            }
        }

        // Diag
        while (diagonalSliders != 0) {
            int startSquare = BitBoardUtility::PopLSB(diagonalSliders);
            unsigned long long moveSquares =
                Magic::GetBishopAttacks(startSquare, allPieces) & moveMask;

            // If piece is pinned, it can only move along the pin ray
            if (IsPinned(startSquare)) {
                moveSquares &= PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare];
            }

            while (moveSquares != 0) {
                int targetSquare = BitBoardUtility::PopLSB(moveSquares);
                moves[currMoveIndex++] = Move(startSquare, targetSquare);
            }
        }
    }

    void GenerateKnightMoves(std::vector<Move> &moves) {
        int friendlyKnightPiece = Piece::MakePiece(Piece::Knight, board->MoveColour());
        // bitboard of all non-pinned knights
        unsigned long long knights = board->PieceBitboards[friendlyKnightPiece] & notPinRays;
        unsigned long long moveMask = emptyOrEnemySquares & checkRayBitmask & moveTypeMask;

        while (knights != 0) {
            int knightSquare = BitBoardUtility::PopLSB(knights);
            unsigned long long moveSquares =
                BitBoardUtility::KnightAttacks[knightSquare] & moveMask;

            while (moveSquares != 0) {
                int targetSquare = BitBoardUtility::PopLSB(moveSquares);
                moves[currMoveIndex++] = Move(knightSquare, targetSquare);
            }
        }
    }

    void GeneratePawnMoves(std::vector<Move> &moves) {
        int pushDir = board->IsWhiteToMove ? 1 : -1;
        int pushOffset = pushDir * 8;

        int friendlyPawnPiece = Piece::MakePiece(Piece::Pawn, board->MoveColour());
        unsigned long long pawns = board->PieceBitboards[friendlyPawnPiece];

        unsigned long long promotionRankMask =
            board->IsWhiteToMove ? BitBoardUtility::Rank8 : BitBoardUtility::Rank1;

        unsigned long long singlePush = (BitBoardUtility::Shift(pawns, pushOffset)) & emptySquares;

        unsigned long long pushPromotions = singlePush & promotionRankMask & checkRayBitmask;

        unsigned long long captureEdgeFileMask =
            board->IsWhiteToMove ? BitBoardUtility::notAFile : BitBoardUtility::notHFile;
        unsigned long long captureEdgeFileMask2 =
            board->IsWhiteToMove ? BitBoardUtility::notHFile : BitBoardUtility::notAFile;
        unsigned long long captureA =
            BitBoardUtility::Shift(pawns & captureEdgeFileMask, pushDir * 7) & enemyPieces;
        unsigned long long captureB =
            BitBoardUtility::Shift(pawns & captureEdgeFileMask2, pushDir * 9) & enemyPieces;

        unsigned long long singlePushNoPromotions =
            singlePush & ~promotionRankMask & checkRayBitmask;

        unsigned long long capturePromotionsA = captureA & promotionRankMask & checkRayBitmask;
        unsigned long long capturePromotionsB = captureB & promotionRankMask & checkRayBitmask;

        captureA &= checkRayBitmask & ~promotionRankMask;
        captureB &= checkRayBitmask & ~promotionRankMask;

        // Single / double push
        if (generateQuietMoves) {
            // Generate single pawn pushes
            while (singlePushNoPromotions != 0) {
                int targetSquare = BitBoardUtility::PopLSB(singlePushNoPromotions);
                int startSquare = targetSquare - pushOffset;
                if (!IsPinned(startSquare) ||
                    PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare] ==
                        PrecomputedMoveData::alignMask[targetSquare][friendlyKingSquare]) {
                    moves[currMoveIndex++] = Move(startSquare, targetSquare);
                }
            }

            // Generate double pawn pushes
            unsigned long long doublePushTargetRankMask =
                board->IsWhiteToMove ? BitBoardUtility::Rank4 : BitBoardUtility::Rank5;
            unsigned long long doublePush = BitBoardUtility::Shift(singlePush, pushOffset) &
                                            emptySquares & doublePushTargetRankMask &
                                            checkRayBitmask;

            while (doublePush != 0) {
                int targetSquare = BitBoardUtility::PopLSB(doublePush);
                int startSquare = targetSquare - pushOffset * 2;
                if (!IsPinned(startSquare) ||
                    PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare] ==
                        PrecomputedMoveData::alignMask[targetSquare][friendlyKingSquare]) {
                    moves[currMoveIndex++] = Move(startSquare, targetSquare, Move::PawnTwoUpFlag);
                }
            }
        }

        // Captures
        while (captureA != 0) {
            int targetSquare = BitBoardUtility::PopLSB(captureA);
            int startSquare = targetSquare - pushDir * 7;

            if (!IsPinned(startSquare) ||
                PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare] ==
                    PrecomputedMoveData::alignMask[targetSquare][friendlyKingSquare]) {
                moves[currMoveIndex++] = Move(startSquare, targetSquare);
            }
        }

        while (captureB != 0) {
            int targetSquare = BitBoardUtility::PopLSB(captureB);
            int startSquare = targetSquare - pushDir * 9;

            if (!IsPinned(startSquare) ||
                PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare] ==
                    PrecomputedMoveData::alignMask[targetSquare][friendlyKingSquare]) {
                moves[currMoveIndex++] = Move(startSquare, targetSquare);
            }
        }

        // Promotions
        while (pushPromotions != 0) {
            int targetSquare = BitBoardUtility::PopLSB(pushPromotions);
            int startSquare = targetSquare - pushOffset;
            if (!IsPinned(startSquare)) {
                GeneratePromotions(startSquare, targetSquare, moves);
            }
        }

        while (capturePromotionsA != 0) {
            int targetSquare = BitBoardUtility::PopLSB(capturePromotionsA);
            int startSquare = targetSquare - pushDir * 7;

            if (!IsPinned(startSquare) ||
                PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare] ==
                    PrecomputedMoveData::alignMask[targetSquare][friendlyKingSquare]) {
                GeneratePromotions(startSquare, targetSquare, moves);
            }
        }

        while (capturePromotionsB != 0) {
            int targetSquare = BitBoardUtility::PopLSB(capturePromotionsB);
            int startSquare = targetSquare - pushDir * 9;

            if (!IsPinned(startSquare) ||
                PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare] ==
                    PrecomputedMoveData::alignMask[targetSquare][friendlyKingSquare]) {
                GeneratePromotions(startSquare, targetSquare, moves);
            }
        }

        // En passant
        if (board->CurrentGameState.enPassantFile > 0) {
            int epFileIndex = board->CurrentGameState.enPassantFile - 1;
            int epRankIndex = board->IsWhiteToMove ? 5 : 2;
            int targetSquare = epRankIndex * 8 + epFileIndex;
            int capturedPawnSquare = targetSquare - pushOffset;

            if (BitBoardUtility::ContainsSquare(checkRayBitmask, capturedPawnSquare)) {
                unsigned long long pawnsThatCanCaptureEp =
                    pawns &
                    BitBoardUtility::PawnAttacks(1ul << targetSquare, !board->IsWhiteToMove);

                while (pawnsThatCanCaptureEp != 0) {
                    int startSquare = BitBoardUtility::PopLSB(pawnsThatCanCaptureEp);
                    if (!IsPinned(startSquare) ||
                        PrecomputedMoveData::alignMask[startSquare][friendlyKingSquare] ==
                            PrecomputedMoveData::alignMask[targetSquare][friendlyKingSquare]) {
                        if (!InCheckAfterEnPassant(startSquare, targetSquare, capturedPawnSquare)) {
                            moves[currMoveIndex++] =
                                Move(startSquare, targetSquare, Move::EnPassantCaptureFlag);
                        }
                    }
                }
            }
        }
    }

    void GeneratePromotions(int startSquare, int targetSquare, std::vector<Move> &moves) {
        moves[currMoveIndex++] = Move(startSquare, targetSquare, Move::PromoteToQueenFlag);
        // Don't generate non-queen promotions in q-search
        if (generateQuietMoves) {
            if (promotionsToGenerate == MoveGenerator::PromotionMode::All) {
                moves[currMoveIndex++] = Move(startSquare, targetSquare, Move::PromoteToKnightFlag);
                moves[currMoveIndex++] = Move(startSquare, targetSquare, Move::PromoteToRookFlag);
                moves[currMoveIndex++] = Move(startSquare, targetSquare, Move::PromoteToBishopFlag);
            } else if (promotionsToGenerate == MoveGenerator::PromotionMode::QueenAndKnight) {
                moves[currMoveIndex++] = Move(startSquare, targetSquare, Move::PromoteToKnightFlag);
            }
        }
    }

    bool IsPinned(int square) const { return ((pinRays >> square) & 1) != 0; }

    void UpdateSlideAttack(unsigned long long pieceBoard, bool ortho) {
        unsigned long long blockers = board->AllPiecesBitboard & ~(1ul << friendlyKingSquare);

        while (pieceBoard != 0) {
            int startSquare = BitBoardUtility::PopLSB(pieceBoard);
            unsigned long long moveBoard = Magic::GetSliderAttacks(startSquare, blockers, ortho);

            opponentSlidingAttackMap |= moveBoard;
        }
    }

    void GenSlidingAttackMap() {
        opponentSlidingAttackMap = 0;

        UpdateSlideAttack(board->EnemyOrthogonalSliders, true);
        UpdateSlideAttack(board->EnemyDiagonalSliders, false);
    }

    void CalculateAttackData() {
        GenSlidingAttackMap();
        // Search squares in all directions around friendly king for checks/pins by enemy sliding
        // pieces (queen, rook, bishop)
        int startDirIndex = 0;
        int endDirIndex = 8;

        if (board->Queens[enemyIndex].Count() == 0) {
            startDirIndex = (board->Rooks[enemyIndex].Count() > 0) ? 0 : 4;
            endDirIndex = (board->Bishops[enemyIndex].Count() > 0) ? 8 : 4;
        }

        for (int dir = startDirIndex; dir < endDirIndex; dir++) {
            bool isDiagonal = dir > 3;
            unsigned long long slider =
                isDiagonal ? board->EnemyDiagonalSliders : board->EnemyOrthogonalSliders;
            if ((PrecomputedMoveData::dirRayMask[dir][friendlyKingSquare] & slider) == 0) {
                continue;
            }

            int n = PrecomputedMoveData::numSquaresToEdge[friendlyKingSquare][dir];
            int directionOffset = PrecomputedMoveData::directionOffsets[dir];
            bool isFriendlyPieceAlongRay = false;
            unsigned long long rayMask = 0;

            for (int i = 0; i < n; i++) {
                int squareIndex = friendlyKingSquare + directionOffset * (i + 1);
                rayMask |= 1ul << squareIndex;
                int piece = board->Square[squareIndex];

                // This square contains a piece
                if (piece != Piece::None) {
                    if (Piece::IsColour(piece, friendlyColour)) {
                        // First friendly piece we have come across in this direction, so it might
                        // be pinned
                        if (!isFriendlyPieceAlongRay) {
                            isFriendlyPieceAlongRay = true;
                        }
                        // This is the second friendly piece we've found in this direction,
                        // therefore pin is not possible
                        else {
                            break;
                        }
                    }
                    // This square contains an enemy piece
                    else {
                        int pieceType = Piece::PieceType(piece);

                        // Check if piece is in bitmask of pieces able to move in current direction
                        if (isDiagonal && Piece::IsDiagonalSlider(pieceType) ||
                            !isDiagonal && Piece::IsOrthogonalSlider(pieceType)) {
                            // Friendly piece blocks the check, so this is a pin
                            if (isFriendlyPieceAlongRay) {
                                pinRays |= rayMask;
                            }
                            // No friendly piece blocking the attack, so this is a check
                            else {
                                checkRayBitmask |= rayMask;
                                inDoubleCheck =
                                    inCheck; // if already in check, then this is double check
                                inCheck = true;
                            }
                            break;
                        } else {
                            // This enemy piece is not able to move in the current direction, and so
                            // is blocking any checks/pins
                            break;
                        }
                    }
                }
            }
            // Stop searching for pins if in double check, as the king is the only piece able to
            // move in that case anyway
            if (inDoubleCheck) {
                break;
            }
        }

        notPinRays = ~pinRays;

        unsigned long long opponentKnightAttacks = 0;
        unsigned long long knights =
            board->PieceBitboards[Piece::MakePiece(Piece::Knight, board->OpponentColour())];
        unsigned long long friendlyKingBoard =
            board->PieceBitboards[Piece::MakePiece(Piece::King, board->MoveColour())];

        while (knights != 0) {
            int knightSquare = BitBoardUtility::PopLSB(knights);
            unsigned long long knightAttacks = BitBoardUtility::KnightAttacks[knightSquare];
            opponentKnightAttacks |= knightAttacks;

            if ((knightAttacks & friendlyKingBoard) != 0) {
                inDoubleCheck = inCheck;
                inCheck = true;
                checkRayBitmask |= 1ul << knightSquare;
            }
        }

        // Pawn attacks
        PieceList opponentPawns = board->Pawns[enemyIndex];
        opponentPawnAttackMap = 0;

        unsigned long long opponentPawnsBoard =
            board->PieceBitboards[Piece::MakePiece(Piece::Pawn, board->OpponentColour())];
        opponentPawnAttackMap = BitBoardUtility::PawnAttacks(opponentPawnsBoard, !isWhiteToMove);
        if (BitBoardUtility::ContainsSquare(opponentPawnAttackMap, friendlyKingSquare)) {
            inDoubleCheck = inCheck; // if already in check, then this is double check
            inCheck = true;
            unsigned long long possiblePawnAttackOrigins =
                board->IsWhiteToMove ? BitBoardUtility::WhitePawnAttacks[friendlyKingSquare]
                                     : BitBoardUtility::BlackPawnAttacks[friendlyKingSquare];
            unsigned long long pawnCheckMap = opponentPawnsBoard & possiblePawnAttackOrigins;
            checkRayBitmask |= pawnCheckMap;
        }

        int enemyKingSquare = board->KingSquare[enemyIndex];

        opponentAttackMapNoPawns = opponentSlidingAttackMap | opponentKnightAttacks |
                                   BitBoardUtility::KingMoves[enemyKingSquare];
        opponentAttackMap = opponentAttackMapNoPawns | opponentPawnAttackMap;

        if (!inCheck) {
            checkRayBitmask = std::numeric_limits<unsigned long long>::max();
        }
    }

    // Test if capturing a pawn with en-passant reveals a sliding piece attack against the king
    // Note: this is only used for cases where pawn appears to not be pinned due to opponent pawn
    // being on same rank (therefore only need to check orthogonal sliders)
    bool InCheckAfterEnPassant(int startSquare, int targetSquare, int epCaptureSquare) {
        unsigned long long enemyOrtho = board->EnemyOrthogonalSliders;

        if (enemyOrtho != 0) {
            unsigned long long maskedBlockers =
                (allPieces ^ (1ul << epCaptureSquare | 1ul << startSquare | 1ul << targetSquare));
            unsigned long long rookAttacks =
                Magic::GetRookAttacks(friendlyKingSquare, maskedBlockers);
            return (rookAttacks & enemyOrtho) != 0;
        }

        return false;
    }
};

} // namespace Chess