#pragma once

#include "common.hpp"

#include "MoveEncoding.hpp"
#include "SelfPlay.hpp"
#include "SelfPlayGame.hpp"
#include "TrainingStats.hpp"

#include "AlphaZeroBase.hpp"

class AlphaZeroSelfPlayer : AlphaZeroBase {
public:
    AlphaZeroSelfPlayer(Network &model, const TrainingArgs &args)
        : AlphaZeroBase(model, args) {}

    void run() {
        SelfPlay selfPlay(m_model, m_args);
        SelfPlayStats selfPlayStats;
        m_model->eval(); // Set model to evaluation mode

        for (size_t iteration = m_startingIteration; true; ++iteration) {
            log("Self Play Iteration", (iteration + 1));

            // Collect new memories from self-play
            selfPlayStats += timeit([&] { return selfPlay.selfPlay(); }, "selfPlay");

            // Output stats
            log("Timeit stats:\n", getTimeitResults());
            log("Self Play Stats:\n", selfPlayStats.toString());

            loadLatestModel(); // Synchronize model across nodes/instances
        }
    }
};
