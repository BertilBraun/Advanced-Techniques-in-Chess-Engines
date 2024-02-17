#pragma once

#include "BoardHelper.h"
#include "Engine/Board/Board.h"
#include "Engine/Board/Coord.h"
#include "Engine/Board/Move.h"
#include "Engine/Board/Piece.h"
#include "Engine/Move Generation/MoveGenerator.h"
#include <cctype>
#include <cmath>
#include <memory>
#include <string>

namespace Chess {
class MoveUtility final {
    /// <summary>
    /// Converts a moveName into internal move representation
    /// Name is expected in format: "e2e4"
    /// Promotions can be written with or without equals sign, for example: "e7e8=q" or "e7e8q"
    /// </summary>
public:
    static Move GetMoveFromUCIName(const std::string &moveName, const Board &board) {
        int startSquare = BoardHelper::SquareIndexFromName(moveName.substr(0, 2));
        int targetSquare = BoardHelper::SquareIndexFromName(moveName.substr(2, 2));

        int movedPieceType = Piece::PieceType(board.Square[startSquare]);
        Coord startCoord = Coord(startSquare);
        Coord targetCoord = Coord(targetSquare);

        // Figure out move flag
        int flag = Move::NoFlag;

        if (movedPieceType == Piece::Pawn) {
            // Promotion
            if (moveName.length() > 4) {
                char lastChar = moveName.back();
                switch (lastChar) {
                case 'q':
                    flag = Move::PromoteToQueenFlag;
                    break;
                case 'r':
                    flag = Move::PromoteToRookFlag;
                    break;
                case 'b':
                    flag = Move::PromoteToBishopFlag;
                    break;
                case 'n':
                    flag = Move::PromoteToKnightFlag;
                    break;
                default:
                    throw std::exception("Invalid promotion piece");
                }
            }
            // Double pawn push
            else if (std::abs(targetCoord.rankIndex - startCoord.rankIndex) == 2) {
                flag = Move::PawnTwoUpFlag;
            }
            // En-passant
            else if (startCoord.fileIndex != targetCoord.fileIndex &&
                     board.Square[targetSquare] == Piece::None) {
                flag = Move::EnPassantCaptureFlag;
            }
        } else if (movedPieceType == Piece::King) {
            if (std::abs(startCoord.fileIndex - targetCoord.fileIndex) > 1) {
                flag = Move::CastleFlag;
            }
        }

        return Move(startSquare, targetSquare, flag);
    }

    /// <summary>
    /// Get algebraic name of move (with promotion specified)
    /// Examples: "e2e4", "e7e8q"
    /// </summary>
    static std::string GetMoveNameUCI(Move move) {
        std::string startSquareName = BoardHelper::SquareNameFromIndex(move.StartSquare());
        std::string endSquareName = BoardHelper::SquareNameFromIndex(move.TargetSquare());
        std::string moveName = startSquareName + endSquareName;
        if (move.IsPromotion()) {
            switch (move.MoveFlag()) {
            case Move::PromoteToRookFlag:
                moveName += "r";
                break;
            case Move::PromoteToKnightFlag:
                moveName += "n";
                break;
            case Move::PromoteToBishopFlag:
                moveName += "b";
                break;
            case Move::PromoteToQueenFlag:
                moveName += "q";
                break;
            }
        }
        return moveName;
    }

    /// <summary>
    /// Get name of move in Standard Algebraic Notation (SAN)
    /// Examples: "e4", "Bxf7+", "O-O", "Rh8#", "Nfd2"
    /// Note, the move must not yet have been made on the board
    /// </summary>
    /*static std::string GetMoveNameSAN(Move move, Board &board) {
        if (move.IsNull()) {
            return "Null";
        }
        int movePieceType = Piece::PieceType(board.Square[move.StartSquare()]);
        int capturedPieceType = Piece::PieceType(board.Square[move.TargetSquare()]);

        if (move.MoveFlag() == Move::CastleFlag) {
            int delta = move.TargetSquare() - move.StartSquare();
            if (delta == 2) {
                return "O-O";
            } else if (delta == -2) {
                return "O-O-O";
            }
        }

        MoveGenerator moveGen;
        std::string moveNotation = std::toupper(Piece::GetSymbol(movePieceType)) + "";

        // check if any ambiguity exists in notation (e.g if e2 can be reached via Nfe2 and Nbe2)
        if (movePieceType != Piece::Pawn && movePieceType != Piece::King) {
            auto allMoves = moveGen.GenerateMoves(board);

            for (auto altMove : allMoves) {

                if (altMove.StartSquare() != move.StartSquare() &&
                    altMove.TargetSquare() ==
                        move.TargetSquare()) { // if moving to same square from different square
                    if (Piece::PieceType(board.Square[altMove.StartSquare()]) ==
                        movePieceType) { // same piece type
                        int fromFileIndex = BoardHelper::FileIndex(move.StartSquare());
                        int alternateFromFileIndex = BoardHelper::FileIndex(altMove.TargetSquare());
                        int fromRankIndex = BoardHelper::RankIndex(move.StartSquare());
                        int alternateFromRankIndex = BoardHelper::RankIndex(altMove.StartSquare());

                        if (fromFileIndex !=
                            alternateFromFileIndex) { // pieces on different files, thus ambiguity
                                                      // can be resolved by specifying file
                            moveNotation += BoardHelper::fileNames[fromFileIndex];
                            break; // ambiguity resolved
                        } else if (fromRankIndex != alternateFromRankIndex) {
                            moveNotation += BoardHelper::rankNames[fromRankIndex];
                            break; // ambiguity resolved
                        }
                    }
                }
            }
        }

        if (capturedPieceType != 0) {
            // add 'x' to indicate capture
            if (movePieceType == Piece::Pawn) {
                moveNotation += BoardHelper::fileNames[BoardHelper::FileIndex(move.StartSquare())];
            }
            moveNotation += "x";
        } else {
            // check if capturing ep
            if (move.MoveFlag() == Move::EnPassantCaptureFlag) {
                moveNotation +=
                    StringHelper::toString(
                        BoardHelper::fileNames[BoardHelper::FileIndex(move.StartSquare())]) +
                    "x";
            }
        }

        moveNotation += BoardHelper::fileNames[BoardHelper::FileIndex(move.TargetSquare())];
        moveNotation += BoardHelper::rankNames[BoardHelper::RankIndex(move.TargetSquare())];

        // add promotion piece
        if (move.IsPromotion()) {
            int promotionPieceType = move.PromotionPieceType();
            moveNotation += "=" + std::toupper(Piece::GetSymbol(promotionPieceType));
        }

        board.MakeMove(move, inSearch : true);
        auto legalResponses = moveGen.GenerateMoves(board);
        // add check/mate symbol if applicable
        if (moveGen.InCheck()) {
            if (legalResponses.size() == 0) {
                moveNotation += "#";
            } else {
                moveNotation += "+";
            }
        }
        board.UnmakeMove(move, inSearch : true);

        return moveNotation;
    }*/

    /// <summary>
    /// Get move from the given name in SAN notation (e.g. "Nxf3", "Rad1", "O-O", etc.)
    /// The given board must contain the position from before the move was made
    /// </summary>
    /*static Move GetMoveFromSAN(const Board &board, const std::string &algebraicMove) {
        MoveGenerator moveGenerator;

        // Remove un-required info from move string
        algebraicMove = StringHelper::replace(
            StringHelper::replace(
                StringHelper::replace(StringHelper::replace(algebraicMove, "+", ""), "#", ""), "x",
                ""),
            "-", "");
        auto allMoves = moveGenerator.GenerateMoves(board);

        Move move = Move();

        for (auto moveToTest : allMoves) {
            move = moveToTest;

            int moveFromIndex = move.StartSquare();
            int moveToIndex = move.TargetSquare();
            int movePieceType = Piece::PieceType(board.Square[moveFromIndex]);
            Coord fromCoord = BoardHelper::CoordFromIndex(moveFromIndex);
            Coord toCoord = BoardHelper::CoordFromIndex(moveToIndex);
            if (algebraicMove == "OO") {
                // castle kingside
                if (movePieceType == Piece::King && moveToIndex - moveFromIndex == 2) {
                    return move;
                }
            } else if (algebraicMove == "OOO") {
                // castle queenside
                if (movePieceType == Piece::King && moveToIndex - moveFromIndex == -2) {
                    return move;
                }
            }
            // Is pawn move if starts with any file indicator (e.g. 'e'4. Note that uppercase B is
            // used for bishops)
            else if (BoardHelper::fileNames.contains(algebraicMove[0].ToString())) {
                if (movePieceType != Piece::Pawn) {
                    continue;
                }
                if (BoardHelper::fileNames.IndexOf(algebraicMove[0]) == fromCoord.fileIndex) {
                    // correct starting file
                    if (algebraicMove.contains('=')) {
                        // is promotion
                        if (toCoord.rankIndex == 0 || toCoord.rankIndex == 7) {

                            if (algebraicMove.size() == 5) // pawn is capturing to promote
                            {
                                char targetFile = algebraicMove[1];
                                if (BoardHelper::fileNames.IndexOf(targetFile) !=
                                    toCoord.fileIndex) {
                                    // Skip if not moving to correct file
                                    continue;
                                }
                            }
                            char promotionChar = algebraicMove[algebraicMove.size() - 1];

                            if (move.PromotionPieceType() !=
                                Piece::GetPieceTypeFromSymbol(promotionChar)) {
                                continue; // skip this move, incorrect promotion type
                            }

                            return move;
                        }
                    } else {
                        char targetFile = algebraicMove[algebraicMove.size() - 2];
                        char targetRank = algebraicMove[algebraicMove.size() - 1];

                        if (BoardHelper::fileNames.IndexOf(targetFile) == toCoord.fileIndex) {
                            // correct ending file
                            if (targetRank.ToString() == (toCoord.rankIndex + 1).ToString()) {
                                // correct ending rank
                                break;
                            }
                        }
                    }
                }
            } else {
                // regular piece move
                char movePieceChar = algebraicMove[0];
                if (Piece::GetPieceTypeFromSymbol(movePieceChar) != movePieceType) {
                    continue; // skip this move, incorrect move piece type
                }

                char targetFile = algebraicMove[algebraicMove.size() - 2];
                char targetRank = algebraicMove[algebraicMove.size() - 1];
                if (BoardHelper::fileNames.IndexOf(targetFile) == toCoord.fileIndex) {
                    // correct ending file
                    if (targetRank.ToString() == (toCoord.rankIndex + 1).ToString()) {
                        // correct ending rank
                        if (algebraicMove.size() == 4) {
                            // addition char present for disambiguation (e.g. Nbd7 or R7e2)
                            char disambiguationChar = algebraicMove[1];

                            if (BoardHelper::fileNames.contains(disambiguationChar.ToString())) {
                                // is file disambiguation
                                if (BoardHelper::fileNames.IndexOf(disambiguationChar) !=
                                    fromCoord.fileIndex) {
                                    // incorrect starting file
                                    continue;
                                }
                            } else {
                                // is rank disambiguation
                                if (disambiguationChar.ToString() !=
                                    (fromCoord.rankIndex + 1).ToString()) {
                                    // incorrect starting rank
                                    continue;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
        return move;
    }*/
};
} // namespace Chess