
// The reason for this build is, that the C++ and Python implementations of PyTorch are apparently
// not really compatible with each other. The Python implementation is apparently not able to load
// the C++ implementation of the model. Therefore, we have to use the C++ implementation of the
// model to load the model and then use the C++ implementation of the model to evaluate the model.
// This is the reason for this build.

#include "common.hpp"

#include "AlphaMCTSNode.hpp"
#include "AlphaZeroBase.hpp"
#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"
#include "Network.hpp"

void iterate(Network &model, AlphaMCTSNode *node);

std::pair<std::vector<std::pair<Move, float>>, float> inference(Network &model, Board &board);

bool hasTimeElapsed(std::chrono::time_point<std::chrono::high_resolution_clock> startTime,
                    int timeToThinkMs) {
    auto currentTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() >=
           timeToThinkMs;
}

Network loadModel() {
    Network model;
    TrainingArgs args;
    args.savePath = "models";
    AlphaZeroBase alphaZeroBase(model, args);
    model->eval();
    return model;
}

int main(int argc, char *argv[]) {

    // This module can be called in multiple ways:
    // 1. "eval" -> Evaluate a "position" using the model, which prints the position evaluation to
    //              the console and afterwards prints the moves with their evaluations to the
    //              console.
    // 2. "play" -> Given a "position" and a "time to think (ms)", the model will search for the
    //              best move and print it to the console.

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <eval|play> [args...]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "eval") {
        // Evaluate a position using the model
        // The position is given as a FEN string
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " eval <position>" << std::endl;
            return 1;
        }

        std::string position = argv[2];

        Network model = loadModel();
        Board board = Board::fromFEN(position);

        auto [moves, value] = inference(model, board);

        std::cout << "Position evaluation: " << value << std::endl;
        std::cout << "Moves:" << std::endl;
        for (auto [move, probability] : moves) {
            std::cout << move.uci() << " " << probability << std::endl;
        }
    } else if (mode == "play") {
        // Play a position using the model
        // The position is given as a FEN string
        // The time to think is given in milliseconds
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " play <position> <time_to_think>" << std::endl;
            return 1;
        }

        std::string position = argv[2];
        int timeToThinkMs = std::stoi(argv[3]);

        Network model = loadModel();
        Board board = Board::fromFEN(position);

        AlphaMCTSNode *root = AlphaMCTSNode::root(board);

        auto startTime = std::chrono::high_resolution_clock::now();

        while (!hasTimeElapsed(startTime, timeToThinkMs)) {
            iterate(model, root);
        }

        auto bestChild = root->bestChild();
        std::cout << "Best move: " << bestChild.move_to_get_here.uci() << std::endl;
        std::cout << "Number of visits: " << bestChild.number_of_visits << std::endl;
        std::cout << "Result Score: " << bestChild.result_score << std::endl;
        std::cout << "Policy: " << bestChild.policy << std::endl;
    } else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}

void iterate(Network &model, AlphaMCTSNode *node) {
    auto current = node;
    while (!current->isTerminalNode()) {
        if (current && !current->isFullyExpanded()) {
            auto [moves, value] = inference(model, current->board);
            current->expand(moves);
            current->backPropagate(value);
            return;
        } else {
            current = &current->bestChild();
        }
    }
}

std::pair<std::vector<std::pair<Move, float>>, float> inference(Network &model, Board &board) {
    torch::NoGradGuard no_grad; // Disable gradient calculation equivalent to torch.no_grad()

    auto [policy, value] = model->inference(encodeBoards({board}).to(model->device));

    auto moves = filterPolicyThenGetMovesAndProbabilities(policy[0], board);

    return std::make_pair(moves, value[0].item<float>());
}