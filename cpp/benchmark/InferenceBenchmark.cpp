#include "games/chess/ChessBoard.hpp"
#include "games/chess/ChessEncoding.hpp"
#include "games/chess/ChessGameContract.hpp"
#include "search/InferencePipeline.hpp"
#include "search/SearchInference.hpp"

#include <barrier>
#include <nlohmann/json.hpp>
#include <numeric>

namespace {
constexpr size_t WARMUP_ITERATIONS = 5;

struct Arguments {
    std::string modelPath;
    std::string mode;
    InferenceDevice device = InferenceDevice::Cuda;
    size_t batchSize = 50;
    size_t workers = 1;
    size_t iterations = 20;
    size_t slots = 3;
    uint32_t seed = 0;
};

Arguments parseArguments(const int argumentCount, char **argumentValues) {
    Arguments arguments;
    for (int index = 1; index < argumentCount; index += 2) {
        if (index + 1 >= argumentCount) {
            throw std::invalid_argument("Every benchmark option requires a value");
        }
        const std::string option = argumentValues[index];
        const std::string value = argumentValues[index + 1];
        if (option == "--model") {
            arguments.modelPath = value;
        } else if (option == "--mode") {
            arguments.mode = value;
        } else if (option == "--device") {
            if (value == "cuda") {
                arguments.device = InferenceDevice::Cuda;
            } else if (value == "cpu") {
                arguments.device = InferenceDevice::Cpu;
            } else {
                throw std::invalid_argument("Device must be cpu or cuda");
            }
        } else if (option == "--batch-size") {
            arguments.batchSize = std::stoull(value);
        } else if (option == "--workers") {
            arguments.workers = std::stoull(value);
        } else if (option == "--iterations") {
            arguments.iterations = std::stoull(value);
        } else if (option == "--slots") {
            arguments.slots = std::stoull(value);
        } else if (option == "--seed") {
            arguments.seed = static_cast<uint32_t>(std::stoul(value));
        } else {
            throw std::invalid_argument("Unknown benchmark option: " + option);
        }
    }
    if (arguments.modelPath.empty() || arguments.mode.empty() || arguments.batchSize == 0 ||
        arguments.workers == 0 || arguments.iterations == 0) {
        throw std::invalid_argument(
            "--model, --mode, --batch-size, --workers, and --iterations are required");
    }
    return arguments;
}

std::vector<Board> generateBoards(const size_t count, const uint32_t seed) {
    std::mt19937 randomEngine(seed);
    std::vector<Board> boards;
    boards.reserve(count);
    Board board;
    size_t pliesSinceReset = 0;
    while (boards.size() < count) {
        if (board.isGameOver() || pliesSinceReset >= 160) {
            board = Board();
            pliesSinceReset = 0;
        }
        const std::vector<Stockfish::Move> &moves = board.validMoves();
        std::uniform_int_distribution<size_t> moveDistribution(0, moves.size() - 1);
        board.makeMove(moves[moveDistribution(randomEngine)]);
        ++pliesSinceReset;
        boards.push_back(board);
    }
    return boards;
}

std::vector<CompressedEncodedBoard> encodeBoards(const std::vector<Board> &boards,
                                                 double &elapsedMilliseconds) {
    const auto startedAt = std::chrono::steady_clock::now();
    std::vector<CompressedEncodedBoard> encodings;
    encodings.reserve(boards.size());
    for (const Board &board : boards) {
        encodings.push_back(encodeBoard(board));
    }
    const auto finishedAt = std::chrono::steady_clock::now();
    elapsedMilliseconds = std::chrono::duration<double, std::milli>(finishedAt - startedAt).count();
    return encodings;
}

void fillInput(torch::Tensor &input, const std::vector<CompressedEncodedBoard> &encodings,
               const size_t firstEncoding, const size_t batchSize) {
    std::int8_t *destination = input.data_ptr<std::int8_t>();
    constexpr size_t ENCODED_BOARD_BYTES = ChessRepresentationDimensions::channel_count *
                                           ChessRepresentationDimensions::board_length *
                                           ChessRepresentationDimensions::board_length;
    for (size_t index = 0; index < batchSize; ++index) {
        writeTensorEncoding(encodings[(firstEncoding + index) % encodings.size()],
                            destination + index * ENCODED_BOARD_BYTES);
    }
}

double outputChecksum(const InferenceOutput &output) {
    return output.policies.sum().item<double>() + output.outcomes.sum().item<double>();
}

nlohmann::json runDirect(const Arguments &arguments,
                         const std::vector<CompressedEncodedBoard> &encodings) {
    InferenceRunner runner(arguments.modelPath, arguments.device, 0, arguments.batchSize, true,
                           ChessGameContract::inferenceDimensions());
    torch::Tensor input = runner.createInputBuffer();
    InferenceOutput output = runner.createOutputBuffer();
    fillInput(input, encodings, 0, arguments.batchSize);
    for (size_t iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        runner.forwardInto(input, arguments.batchSize, output);
    }
    const auto startedAt = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < arguments.iterations; ++iteration) {
        runner.forwardInto(input, arguments.batchSize, output);
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds}, {"checksum", outputChecksum(output)}};
}

nlohmann::json runProcessedDirect(const Arguments &arguments, const std::vector<Board> &boards,
                                  const std::vector<CompressedEncodedBoard> &encodings) {
    InferenceRunner runner(arguments.modelPath, arguments.device, 0, arguments.batchSize, true,
                           ChessGameContract::inferenceDimensions());
    torch::Tensor input = runner.createInputBuffer();
    InferenceOutput output = runner.createOutputBuffer();
    fillInput(input, encodings, 0, arguments.batchSize);
    for (size_t iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        runner.forwardInto(input, arguments.batchSize, output);
    }

    double checksum = 0.0;
    const auto startedAt = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < arguments.iterations; ++iteration) {
        runner.forwardInto(input, arguments.batchSize, output);
        const float *policyData = output.policies.data_ptr<float>();
        const float *outcomeData = output.outcomes.data_ptr<float>();
        for (size_t position = 0; position < arguments.batchSize; ++position) {
            const SearchInferenceResult<ChessGameContract> result =
                processSearchInference<ChessGameContract>(
                    policyData + position * ChessAction::action_count, outcomeData + position * 3,
                    boards[position]);
            checksum += result.value() + static_cast<double>(result.actions.size());
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds}, {"checksum", checksum}};
}

nlohmann::json runPipeline(const Arguments &arguments,
                           const std::vector<CompressedEncodedBoard> &encodings) {
    InferencePipeline pipeline(arguments.modelPath, arguments.device, 0, arguments.batchSize,
                               arguments.slots, true, ChessGameContract::inferenceDimensions());
    std::vector<size_t> pendingSlots;
    pendingSlots.reserve(arguments.slots);
    constexpr size_t ENCODED_BOARD_BYTES = ChessRepresentationDimensions::channel_count *
                                           ChessRepresentationDimensions::board_length *
                                           ChessRepresentationDimensions::board_length;
    for (size_t warmupIteration = 0; warmupIteration < WARMUP_ITERATIONS; ++warmupIteration) {
        const InferencePipeline::WritableBatch warmup = pipeline.acquireWritableBatch();
        for (size_t index = 0; index < arguments.batchSize; ++index) {
            writeTensorEncoding(encodings[index % encodings.size()],
                                warmup.data + index * ENCODED_BOARD_BYTES);
        }
        pipeline.submit(warmup.slotIndex, arguments.batchSize);
        static_cast<void>(pipeline.waitCompleted(warmup.slotIndex));
        pipeline.release(warmup.slotIndex);
    }
    const auto startedAt = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < arguments.iterations; ++iteration) {
        if (pendingSlots.size() == arguments.slots) {
            static_cast<void>(pipeline.waitCompleted(pendingSlots.front()));
            pipeline.release(pendingSlots.front());
            pendingSlots.erase(pendingSlots.begin());
        }
        const InferencePipeline::WritableBatch writable = pipeline.acquireWritableBatch();
        for (size_t index = 0; index < arguments.batchSize; ++index) {
            writeTensorEncoding(
                encodings[(iteration * arguments.batchSize + index) % encodings.size()],
                writable.data + index * ENCODED_BOARD_BYTES);
        }
        pipeline.submit(writable.slotIndex, arguments.batchSize);
        pendingSlots.push_back(writable.slotIndex);
    }
    InferenceOutput finalOutput;
    for (const size_t slotIndex : pendingSlots) {
        const InferenceOutput output = pipeline.waitCompleted(slotIndex);
        finalOutput = output;
        pipeline.release(slotIndex);
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds}, {"checksum", outputChecksum(finalOutput)}};
}

nlohmann::json runReplicas(const Arguments &arguments,
                           const std::vector<CompressedEncodedBoard> &encodings) {
    std::vector<std::unique_ptr<InferenceRunner>> runners;
    std::vector<torch::Tensor> inputs;
    std::vector<InferenceOutput> outputs;
    runners.reserve(arguments.workers);
    inputs.reserve(arguments.workers);
    outputs.reserve(arguments.workers);
    for (size_t worker = 0; worker < arguments.workers; ++worker) {
        auto runner = std::make_unique<InferenceRunner>(arguments.modelPath, arguments.device, 0,
                                                        arguments.batchSize, true,
                                                        ChessGameContract::inferenceDimensions());
        torch::Tensor input = runner->createInputBuffer();
        InferenceOutput output = runner->createOutputBuffer();
        fillInput(input, encodings, worker * arguments.batchSize, arguments.batchSize);
        for (size_t iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
            runner->forwardInto(input, arguments.batchSize, output);
        }
        inputs.push_back(std::move(input));
        outputs.push_back(std::move(output));
        runners.push_back(std::move(runner));
    }

    std::barrier startBarrier(static_cast<std::ptrdiff_t>(arguments.workers + 1));
    std::vector<std::thread> threads;
    threads.reserve(arguments.workers);
    for (size_t worker = 0; worker < arguments.workers; ++worker) {
        threads.emplace_back([&, worker]() {
            startBarrier.arrive_and_wait();
            for (size_t iteration = 0; iteration < arguments.iterations; ++iteration) {
                runners[worker]->forwardInto(inputs[worker], arguments.batchSize, outputs[worker]);
            }
        });
    }
    startBarrier.arrive_and_wait();
    const auto startedAt = std::chrono::steady_clock::now();
    for (std::thread &thread : threads) {
        thread.join();
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    double checksum = 0.0;
    for (const InferenceOutput &output : outputs) {
        checksum += outputChecksum(output);
    }
    return {{"elapsed_seconds", seconds}, {"checksum", checksum}};
}

nlohmann::json runProcessedReplicas(const Arguments &arguments, const std::vector<Board> &boards,
                                    const std::vector<CompressedEncodedBoard> &encodings) {
    std::vector<std::unique_ptr<InferenceRunner>> runners;
    std::vector<torch::Tensor> inputs;
    std::vector<InferenceOutput> outputs;
    runners.reserve(arguments.workers);
    inputs.reserve(arguments.workers);
    outputs.reserve(arguments.workers);
    for (size_t worker = 0; worker < arguments.workers; ++worker) {
        auto runner = std::make_unique<InferenceRunner>(arguments.modelPath, arguments.device, 0,
                                                        arguments.batchSize, true,
                                                        ChessGameContract::inferenceDimensions());
        torch::Tensor input = runner->createInputBuffer();
        InferenceOutput output = runner->createOutputBuffer();
        fillInput(input, encodings, worker * arguments.batchSize, arguments.batchSize);
        for (size_t iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
            runner->forwardInto(input, arguments.batchSize, output);
        }
        inputs.push_back(std::move(input));
        outputs.push_back(std::move(output));
        runners.push_back(std::move(runner));
    }

    std::barrier startBarrier(static_cast<std::ptrdiff_t>(arguments.workers + 1));
    std::vector<double> checksums(arguments.workers, 0.0);
    std::vector<std::thread> threads;
    threads.reserve(arguments.workers);
    for (size_t worker = 0; worker < arguments.workers; ++worker) {
        threads.emplace_back([&, worker]() {
            startBarrier.arrive_and_wait();
            for (size_t iteration = 0; iteration < arguments.iterations; ++iteration) {
                runners[worker]->forwardInto(inputs[worker], arguments.batchSize, outputs[worker]);
                const float *policyData = outputs[worker].policies.data_ptr<float>();
                const float *outcomeData = outputs[worker].outcomes.data_ptr<float>();
                for (size_t position = 0; position < arguments.batchSize; ++position) {
                    const size_t boardIndex = worker * arguments.batchSize + position;
                    const SearchInferenceResult<ChessGameContract> result =
                        processSearchInference<ChessGameContract>(
                            policyData + position * ChessAction::action_count,
                            outcomeData + position * 3, boards[boardIndex]);
                    checksums[worker] +=
                        result.value() + static_cast<double>(result.actions.size());
                }
            }
        });
    }
    startBarrier.arrive_and_wait();
    const auto startedAt = std::chrono::steady_clock::now();
    for (std::thread &thread : threads) {
        thread.join();
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds},
            {"checksum", std::accumulate(checksums.begin(), checksums.end(), 0.0)}};
}

} // namespace

int main(const int argumentCount, char **argumentValues) {
    try {
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
        const Arguments arguments = parseArguments(argumentCount, argumentValues);
        const size_t positions = arguments.batchSize * arguments.iterations * arguments.workers;
        const std::vector<Board> boards = generateBoards(positions, arguments.seed);
        double encodeMilliseconds = 0.0;
        const std::vector<CompressedEncodedBoard> encodings =
            encodeBoards(boards, encodeMilliseconds);
        constexpr size_t ENCODED_BOARD_BYTES = ChessRepresentationDimensions::channel_count *
                                               ChessRepresentationDimensions::board_length *
                                               ChessRepresentationDimensions::board_length;
        std::vector<std::int8_t> packedEncodings(positions * ENCODED_BOARD_BYTES);
        const auto packingStartedAt = std::chrono::steady_clock::now();
        for (size_t index = 0; index < positions; ++index) {
            writeTensorEncoding(encodings[index],
                                packedEncodings.data() + index * ENCODED_BOARD_BYTES);
        }
        const double packingMilliseconds = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - packingStartedAt)
                                               .count();

        nlohmann::json result;
        if (arguments.mode == "direct") {
            result = runDirect(arguments, encodings);
        } else if (arguments.mode == "processed_direct") {
            result = runProcessedDirect(arguments, boards, encodings);
        } else if (arguments.mode == "pipeline") {
            result = runPipeline(arguments, encodings);
        } else if (arguments.mode == "replicas") {
            result = runReplicas(arguments, encodings);
        } else if (arguments.mode == "processed_replicas") {
            result = runProcessedReplicas(arguments, boards, encodings);
        } else {
            throw std::invalid_argument(
                "Mode must be direct, processed_direct, pipeline, replicas, or "
                "processed_replicas");
        }

        const double elapsedSeconds = result.at("elapsed_seconds").get<double>();
        result["mode"] = arguments.mode;
        result["device"] = arguments.device == InferenceDevice::Cuda ? "cuda" : "cpu";
        result["batch_size"] = arguments.batchSize;
        result["workers"] = arguments.workers;
        result["iterations_per_worker"] = arguments.iterations;
        result["positions"] = positions;
        result["positions_per_second"] = static_cast<double>(positions) / elapsedSeconds;
        result["state_generation_seed"] = arguments.seed;
        result["state_encoding_milliseconds"] = encodeMilliseconds;
        result["state_encoding_positions_per_second"] =
            static_cast<double>(positions) / (encodeMilliseconds / 1000.0);
        result["tensor_packing_milliseconds"] = packingMilliseconds;
        result["tensor_packing_positions_per_second"] =
            static_cast<double>(positions) / (packingMilliseconds / 1000.0);
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
