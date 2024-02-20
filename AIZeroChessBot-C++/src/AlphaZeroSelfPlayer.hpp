#pragma once

#include "common.hpp"

#include "MoveEncoding.hpp"
#include "SelfPlay.hpp"
#include "SelfPlayGame.hpp"
#include "TrainingStats.hpp"

#include "AlphaZeroBase.hpp"

class AlphaZeroSelfPlayer : AlphaZeroBase {
public:
    AlphaZeroSelfPlayer(size_t rank, Network &model, const TrainingArgs &args)
        : AlphaZeroBase(model, args) {}

    void run() {
        SelfPlay selfPlay(m_model, m_args);
        SelfPlayStats selfPlayStats;
        m_model->eval(); // Set model to evaluation mode

        for (size_t iteration = m_startingIteration; true; ++iteration) {
            std::cout << "Self Play Iteration " << (iteration + 1) << std::endl;

            // Collect new memories from self-play
            selfPlayStats += timeit([&] { return selfPlay.selfPlay(); }, "selfPlay");

            // Output stats
            std::cout << "Timeit stats:" << std::endl << get_timeit_results() << std::endl;
            std::cout << "Self Play Stats:" << std::endl << selfPlayStats.toString() << std::endl;

            loadLatestModel(); // Synchronize model across nodes/instances
        }
    }
};
