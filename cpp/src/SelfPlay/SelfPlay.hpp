#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"
#include "MCTS/MCTS.hpp"

#include "SelfPlayGame.hpp"
#include "SelfPlayWriter.hpp"

class SelfPlay {
private:
    SelfPlayWriter *m_writer;
    SelfPlayParams m_args;

    std::vector<SelfPlayGame> m_selfPlayGames;

    MCTS m_mcts;

public:
    SelfPlay(InferenceClient *inferenceClient, SelfPlayWriter *writer, SelfPlayParams args,
             TensorBoardLogger *logger);

    void selfPlay();

private:
    void _handleTooLongGame(const SelfPlayGame &game);

    int _countPieces(const Board &board, Color color) const;

    std::pair<SelfPlayGame, Move> _sampleSPG(const SelfPlayGame &game,
                                             const ActionProbabilities &actionProbabilities);

    Move _sampleMove(int numMoves, const ActionProbabilities &actionProbabilities) const;
};