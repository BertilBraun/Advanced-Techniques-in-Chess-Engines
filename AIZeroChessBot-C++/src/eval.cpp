
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

void uciMainLoop(Network &model, int timeToThinkMs);

Network loadModel();

void iterate(Network &model, AlphaMCTSNode *node);

std::pair<std::vector<std::pair<Move, float>>, float> inference(Network &model, Board &board);

bool hasTimeElapsed(std::chrono::time_point<std::chrono::high_resolution_clock> startTime,
                    int timeToThinkMs);

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

        std::cerr << "Position evaluation: " << value << std::endl;
        std::cerr << "Moves:" << std::endl;
        for (auto [move, probability] : moves) {
            std::cerr << move.uci() << " " << probability << std::endl;
        }
    } else if (mode == "play") {
        // Play a position using the model
        // The position is given as a FEN string
        // The time to think is given in milliseconds
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " play <time_to_think>" << std::endl;
            return 1;
        }

        int timeToThinkMs = std::stoi(argv[2]);
        Network model = loadModel();

        uciMainLoop(model, timeToThinkMs);
    } else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}

void processPositionCommand(const std::string &commandLine);
void processGoCommand(Network &model, int timeToThinkMs);

Board board;

void uciMainLoop(Network &model, int timeToThinkMs) {
    std::string line;
    std::cout << "id name AIZeroChessBot" << std::endl;
    std::cout << "id author Bertil" << std::endl;
    std::cout << "uciok" << std::endl;

    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "uci") {
            std::cout << "id name AIZeroChessBot" << std::endl;
            std::cout << "id author Bertil" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (token == "ucinewgame") {
            board = Board();
        } else if (token == "position") {
            processPositionCommand(line.substr(token.length() + 1));
        } else if (token == "go") {
            processGoCommand(model, timeToThinkMs);
        } else if (token == "quit") {
            break;
        }
        // Implement other commands as needed
    }
}

void processPositionCommand(const std::string &commandLine) {
    std::istringstream iss(commandLine);
    std::string token;
    iss >> token; // Reads "position" keyword which we already know

    if (token == "startpos") {
        board = Board(); // Reset to starting position
        iss >> token;    // Should be "moves" if there are any moves
    } else if (token == "fen") {
        std::string fen;
        while (iss >> token && token != "moves") {
            fen += token + " ";
        }
        board = Board::fromFEN(fen);
    }

    // If there are moves to process
    if (token == "moves") {
        while (iss >> token) {
            board.push(Move::fromUci(token));
        }
    }
}

void processGoCommand(Network &model, int timeToThinkMs) {

    AlphaMCTSNode *root = AlphaMCTSNode::root(board);

    auto startTime = std::chrono::high_resolution_clock::now();

    while (!hasTimeElapsed(startTime, timeToThinkMs)) {
        iterate(model, root);
    }

    auto bestChild = root->bestChild();
    std::cerr << "Best move: " << bestChild.move_to_get_here.uci() << std::endl;
    std::cerr << "Number of visits: " << bestChild.number_of_visits << std::endl;
    std::cerr << "Result Score: " << bestChild.result_score << std::endl;
    std::cerr << "Policy: " << bestChild.policy << std::endl;

    std::cout << "bestmove " << bestChild.move_to_get_here.uci() << std::endl;
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