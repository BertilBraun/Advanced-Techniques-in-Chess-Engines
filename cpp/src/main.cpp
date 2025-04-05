#include "common.hpp"

#include "MCTS/MCTS.hpp"
#include "SelfPlay/SelfPlay.hpp"
#include "SelfPlay/SelfPlayWriter.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic binding of STL containers

void selfPlayMain(int runId, const std::string &savePath, int numProcessors, int numGPUs) {
    assert(runId >= 0);
    assert(numProcessors >= 1);
    assert(0 < numGPUs && numGPUs <= std::max((int) torch::cuda::device_count(), 1));
    assert(std::filesystem::exists(std::filesystem::path(savePath)));
    assert(std::filesystem::is_directory(std::filesystem::path(savePath)));

    const TrainingArgs TRAINING_ARGS = {
        .save_path = savePath,
        .self_play =
            {
                .mcts =
                    {
                        .num_searches_per_turn = 640,
                        .num_parallel_searches = 8,
                        .c_param = 1.7,
                        .dirichlet_alpha = 0.3,
                        .dirichlet_epsilon = 0.25,
                    },
                .num_parallel_games = 32,
                .num_moves_after_which_to_play_greedy = 25,
                .max_moves = 250,
                .result_score_weight = 0.15,
                .resignation_threshold = -1.0,
            },
        .writer =
            {
                .filePrefix = std::string("memory"),
                .batchSize = 5000,
            },
        .inference =
            {
                .maxBatchSize = 128,
            },
    };

    TensorBoardLogger logger(std::string("logs/run_") + std::to_string(runId) +
                             std::string("/tfevents"));

    auto [currentModelPath, currentIteration] =
        get_latest_iteration_save_path(TRAINING_ARGS.save_path);

    std::vector<InferenceClient> clients(numGPUs * 2);
    for (int i : range(numGPUs * 2)) { // start 2 InferenceClients per GPU
        clients[i].init(i % numGPUs, currentModelPath, TRAINING_ARGS.inference.maxBatchSize,
                        &logger);
    }

    SelfPlayWriter writer(TRAINING_ARGS, logger);
    writer.updateIteration(currentIteration);

    log("Number of processors:", numProcessors, "Number of GPUs:", numGPUs);
    log("Starting on run", runId, "with model path:", currentModelPath,
        "Iteration:", currentIteration);

    std::vector<std::thread> threads;
    for (int i : range(numProcessors)) {
        threads.emplace_back(std::thread([&] {
            SelfPlay selfPlay(&clients[i % clients.size()], &writer, TRAINING_ARGS.self_play,
                              &logger);
            log("Worker process", i + 1, "of", numProcessors, "started");

            while (true) {
                selfPlay.selfPlay();
            }
        }));
    }

    while (true) {
        const auto [latestModelPath, latestIteration] =
            get_latest_iteration_save_path(TRAINING_ARGS.save_path);

        if (latestModelPath == currentModelPath) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }
        log("New model found:", latestModelPath, "Iteration:", latestIteration);
        log("Updating model for all clients");
        for (auto &client : clients) {
            client.updateModel(latestModelPath, latestIteration);
        }

        writer.updateIteration(latestIteration);

        reset_times(&logger, currentIteration);

        currentModelPath = latestModelPath;
        currentIteration = latestIteration;
        log("Model updated for all clients");
    }

    log("Main thread finished");
}

typedef std::pair<int, std::vector<std::pair<int, int>>> PyInferenceResult; // (score, visits);

std::vector<PyInferenceResult> boardInferenceMain(const std::string &modelPath,
                                                  const std::vector<std::string> &fens) {
    log("Model path:", modelPath);
    log("Starting board inference");

    const MCTSParams EVALUATION_MCTS_PARAMS = {
        .num_searches_per_turn =
            64, // Reduced number of searches to speed up the evaluation process
        .num_parallel_searches = 8,
        .c_param = 1.7,
        .dirichlet_alpha = 0.3,
        .dirichlet_epsilon = 0.0, // No Dirichlet noise for board inference (greedy)
    };

    const int numBoards = fens.size();
    const int maxBatchSize = numBoards * EVALUATION_MCTS_PARAMS.num_parallel_searches;

    InferenceClient inferenceClient;
    inferenceClient.init(0, modelPath, maxBatchSize, nullptr);
    const MCTS mcts(&inferenceClient, EVALUATION_MCTS_PARAMS, nullptr);

    std::vector<Board> boards;
    boards.reserve(numBoards);
    for (const auto &fen : fens) {
        const Board board = Board::fromFEN(fen);

        if (!board.isValid()) {
            log("Invalid Board fen:", fen);
            throw std::runtime_error("Invalid Board fen: " + fen);
        }
        boards.push_back(board);
    }

    std::vector<PyInferenceResult> pyResults;
    pyResults.reserve(numBoards);

    for (const auto &[score, visits] : mcts.search(boards)) {
        pyResults.emplace_back(score, visits);
    }

    log("Board inference completed");
    return pyResults;
}

Move evalBoardIterate(const std::string &modelPath, const std::string &fen, bool networkOnly,
                      float maxTime) {
    log("Model path:", modelPath);
    log("Starting board inference");
    log("FEN:", fen);
    log("Network only:", networkOnly);
    log("Max time:", maxTime);

    const MCTSParams EVALUATION_MCTS_PARAMS = {
        .num_searches_per_turn =
            2 << 30, // Set to a very high number to ensure we search until maxTime is reached
        .num_parallel_searches = 8,
        .c_param = 1.7,
        .dirichlet_alpha = 0.3,
        .dirichlet_epsilon = 0.0, // No Dirichlet noise for board inference (greedy)
    };

    const int maxBatchSize = 1; // * EVALUATION_MCTS_PARAMS.num_parallel_searches;
    InferenceClient inferenceClient;
    inferenceClient.init(0, modelPath, maxBatchSize, nullptr);
    const MCTS mcts(&inferenceClient, EVALUATION_MCTS_PARAMS, nullptr);

    Board board = Board::fromFEN(fen);
    if (!board.isValid()) {
        log("Invalid Board fen:", fen);
        throw std::runtime_error("Invalid Board fen: " + fen);
    }

    std::vector<Board> boards;
    boards.push_back(board);

    if (networkOnly) {
        // Run the network inference only
        const auto [value, moves] = inferenceClient.inference_batch(boards)[0];
        log("Network inference completed");
        int bestMoveIndex = -1;
        float bestMoveScore = -1.0f;
        for (const auto &[move, score] : moves) {
            if (score > bestMoveScore) {
                bestMoveScore = score;
                bestMoveIndex = move;
            }
        }
        assert(bestMoveIndex != -1);
        log("Best move:", decodeMove(bestMoveIndex), "Best move score:", bestMoveScore);
        log("Result:", value);
        return decodeMove(bestMoveIndex);
    }

    /*

        root = MCTSNode.root(board)

        for _ in range(self.mcts_args.num_searches_per_turn //
       self.mcts_args.num_parallel_searches): self.mcts.parallel_iterate([root]) if self.time_is_up:
                break

        best_move_index = np.argmax(root.children_number_of_visits)
        best_child = root.children[best_move_index]

        def max_depth(node: MCTSNode) -> int:
            if not node.children:
                return 0
            return 1 + max(max_depth(child) for child in node.children)

        log('---------------------- Alpha Zero Best Move ----------------------')
        log('Total number of searches:', root.number_of_visits)
        log('Max depth:', max_depth(root))
        log('Best child index:', best_move_index)
        log(f'Best child has {best_child.number_of_visits} visits')
        log(f'Best child has {best_child.result_score:.4f} result_score')
        log('Child moves:', [child.encoded_move_to_get_here for child in root.children])
        log('Child visits:', root.children_number_of_visits)
        log('Child result_scores:', np.round(root.children_result_scores, 2))
        log('Child priors:', np.round(root.children_policies, 2))
        log('------------------------------------------------------------------')

        return ChessGame.decode_move(best_child.encoded_move_to_get_here)
        */

    // Run MCTS search until maxTime is reached
    auto start = std::chrono::high_resolution_clock::now();
    auto isTimeUp = [&]() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        return duration.count() >= maxTime;
    };

    MCTSNode root = MCTSNode::root(board);

    while (!isTimeUp() && root.number_of_visits < EVALUATION_MCTS_PARAMS.num_searches_per_turn) {
        // Run MCTS search
        mcts.parallel_iterate({root});
    }

    // Get the best move from the MCTS results
    int bestMoveIndex = -1;
    int bestMoveVisits = -1;
    for (size_t i = 0; i < root.children.size(); ++i) {
        const MCTSNode &child = root.children[i];
        if (child.number_of_visits > bestMoveVisits) {
            bestMoveVisits = child.number_of_visits;
            bestMoveIndex = child.encoded_move_to_get_here;
        }
    }
    assert(bestMoveIndex != -1);
    log("Best move:", decodeMove(bestMoveIndex), "Best move visits:", bestMoveVisits);
    log("Result:", root.result_score);
    log("Board inference completed");
    log("Total number of searches:", root.number_of_visits);
    log("Best child visits:", bestMoveVisits);

    return decodeMove(bestMoveIndex);
}

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(AlphaZeroCpp, m) {
    m.doc() = "AlphaZero C++ bindings for Python";

    m.def("self_play_main", &selfPlayMain,
          "Runs self-play on the given model save directory path and parameters", py::arg("run_id"),
          py::arg("save_path"), py::arg("num_processors"), py::arg("num_gpus"))
        .attr("__annotations__") =
        py::dict("return"_a = "None", "run_id"_a = "int", "save_path"_a = "str",
                 "num_processors"_a = "int", "num_gpus"_a = "int");

    m.def("board_inference_main", &boardInferenceMain,
          "Runs board inference on a list of FEN strings", py::arg("model_path"), py::arg("fens"))
        .attr("__annotations__") = py::dict("return"_a = "List[Tuple[int, List[Tuple[int, int]]]]",
                                            "model_path"_a = "str", "fens"_a = "List[str]");

    m.def("eval_board_iterate", &evalBoardIterate, "Runs board inference on a single FEN string",
          py::arg("model_path"), py::arg("fen"), py::arg("network_only") = false,
          py::arg("max_time") = 5.0)
        .attr("__annotations__") =
        py::dict("return"_a = "Move", "model_path"_a = "str", "fen"_a = "str",
                 "network_only"_a = "bool", "max_time"_a = "float");

    // encodeBoard function
    m.def("encode_board", &encodeBoard, "Encodes a board into a CompressedEncodedBoard",
          py::arg("board"));

    py::module_ typing = py::module_::import("typing");
    py::module_ numpy = py::module_::import("numpy");

    // type alias for CompressedEncodedBoard = List[uint64]
    m.attr("CompressedEncodedBoard") = typing.attr("List")(numpy.attr("uint64"));

    // type alias for EncodedBoard = List[List[List[int8]]]
    m.attr("EncodedBoard") =
        typing.attr("List")(typing.attr("List")(typing.attr("List")(numpy.attr("int8"))));

    // compress function
    m.def("compress", &compress, "Compresses an EncodedBoard into a CompressedEncodedBoard",
          py::arg("binary"))
        .attr("__annotations__") =
        py::dict("return"_a = "CompressedEncodedBoard", "binary"_a = "EncodedBoard");

    // decompress function
    m.def("decompress", &decompress, "Decompresses a CompressedEncodedBoard into an EncodedBoard",
          py::arg("compressed"))
        .attr("__annotations__") =
        py::dict("return"_a = "EncodedBoard", "compressed"_a = "CompressedEncodedBoard");

    // register the PieceType enum in Python
    py::enum_<PieceType>(m, "PieceType")
        .value("NONE", PieceType::NONE)
        .value("PAWN", PieceType::PAWN)
        .value("KNIGHT", PieceType::KNIGHT)
        .value("BISHOP", PieceType::BISHOP)
        .value("ROOK", PieceType::ROOK)
        .value("QUEEN", PieceType::QUEEN)
        .value("KING", PieceType::KING);

    // define the Move class in Python with methods to access fromSquare(), toSquare(), and
    // promotion()
    py::class_<Move>(m, "Move")
        .def(py::init<int, int, PieceType>(), py::arg("from_square"), py::arg("to_square"),
             py::arg("promotion") = PieceType::NONE)
        .def("from_square", &Move::fromSquare, "Returns the from square of the move")
        .def("to_square", &Move::toSquare, "Returns the to square of the move")
        .def("promotion", &Move::promotion, "Returns the promotion type of the move")
        .def("__str__", &Move::uci, "Returns the UCI string representation of the move")
        .def("__repr__", [](const Move &move) { return move.uci(); });

    // encodeMove function
    m.def("encode_move", &encodeMove, "Encodes a move into an integer", py::arg("move"))
        .attr("__annotations__") = py::dict("return"_a = "int", "move"_a = "Move");

    // decodeMove function
    m.def("decode_move", &decodeMove, "Decodes an integer into a move", py::arg("move_index"))
        .attr("__annotations__") = py::dict("return"_a = "Move", "move_index"_a = "int");

    // actionProbabilities function
    m.def("action_probabilities", &actionProbabilities,
          "Returns the action probabilities for a given visit counts vector",
          py::arg("visit_counts"))
        .attr("__annotations__") =
        py::dict("return"_a = "List[int]", "visit_counts"_a = "List[Tuple[int, int]]");
}
