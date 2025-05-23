#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"
#include "MCTS/MCTS.hpp"

struct SelfPlayGameMemory {
    Board board;
    VisitCounts visitCounts;
    float result;

    SelfPlayGameMemory(const Board &_board, const VisitCounts &_visitCounts, float _result)
        : board(_board), visitCounts(_visitCounts), result(_result) {}
};

struct SelfPlayGame {
public:
    Board board;
    std::vector<SelfPlayGameMemory> memory;
    std::vector<Move> playedMoves;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;

public:
    SelfPlayGame() : board(Board()), m_startTime(std::chrono::high_resolution_clock::now()) {}

    SelfPlayGame expand(Move move) const {
        SelfPlayGame newGame = this->copy();
        newGame.board.push(move);
        newGame.playedMoves.push_back(move);
        return newGame;
    }

    float generationTime() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - m_startTime);
        return duration.count();
    }

private:
    SelfPlayGame copy() const {
        SelfPlayGame newGame;
        newGame.board = board.copy();
        newGame.memory = memory;
        newGame.playedMoves = playedMoves;
        newGame.m_startTime = m_startTime;
        return newGame;
    }
};
