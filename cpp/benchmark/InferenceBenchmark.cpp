#include "games/chess/ChessGame.hpp"
#include "games/chess/encoding/ChessEncoding.hpp"
#include "games/chess/implementation/ChessBoard.hpp"
#include "search/InferencePipeline.hpp"
#include "util/py.hpp"

#include <ATen/Context.h>
#include <barrier>
#include <nlohmann/json.hpp>
#include <numeric>
#include <span>

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
    std::string sdpaBackend = "automatic";
};

Arguments parseArguments(const int argumentCount, char **argumentValues) {
    Arguments arguments;
    for (const int index : range(1, argumentCount, 2)) {
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
        } else if (option == "--sdpa-backend") {
            arguments.sdpaBackend = value;
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

void configureSdpaBackend(const Arguments &arguments) {
    if (arguments.sdpaBackend == "automatic") {
        return;
    }
    if (arguments.device != InferenceDevice::Cuda) {
        throw std::invalid_argument("An explicit SDPA backend requires CUDA inference");
    }
    const bool useFlash = arguments.sdpaBackend == "flash";
    const bool useMemoryEfficient = arguments.sdpaBackend == "memory_efficient";
    const bool useMath = arguments.sdpaBackend == "math";
    const bool useCuDNN = arguments.sdpaBackend == "cudnn";
    if (!useFlash && !useMemoryEfficient && !useMath && !useCuDNN) {
        throw std::invalid_argument(
            "SDPA backend must be automatic, flash, memory_efficient, math, or cudnn");
    }
    at::globalContext().setSDPUseFlash(useFlash);
    at::globalContext().setSDPUseMemEfficient(useMemoryEfficient);
    at::globalContext().setSDPUseMath(useMath);
    at::globalContext().setSDPUseCuDNN(useCuDNN);
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
    for (const auto index : range(batchSize)) {
        encodings[(firstEncoding + index) % encodings.size()].writeTensorInto(
            std::span(destination + index * ENCODED_BOARD_BYTES, ENCODED_BOARD_BYTES));
    }
}

double outputChecksum(const InferenceOutput &output) {
    return output.policies.sum().item<double>() + output.outcomes.sum().item<double>();
}

nlohmann::json runDirect(const Arguments &arguments,
                         const std::vector<CompressedEncodedBoard> &encodings) {
    InferenceRunner runner(arguments.modelPath, arguments.device, 0, arguments.batchSize, true,
                           ChessGame::Encoding::inferenceDimensions());
    torch::Tensor input = runner.createInputBuffer();
    InferenceOutput output = runner.createOutputBuffer();
    fillInput(input, encodings, 0, arguments.batchSize);
    for (const auto iteration : range(WARMUP_ITERATIONS)) {
        static_cast<void>(iteration);
        runner.forwardInto(input, arguments.batchSize, output);
    }
    const auto startedAt = std::chrono::steady_clock::now();
    for (const auto iteration : range(arguments.iterations)) {
        static_cast<void>(iteration);
        runner.forwardInto(input, arguments.batchSize, output);
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds}, {"checksum", outputChecksum(output)}};
}

nlohmann::json runProcessedDirect(const Arguments &arguments, const std::vector<Board> &boards,
                                  const std::vector<CompressedEncodedBoard> &encodings) {
    InferenceRunner runner(arguments.modelPath, arguments.device, 0, arguments.batchSize, true,
                           ChessGame::Encoding::inferenceDimensions());
    torch::Tensor input = runner.createInputBuffer();
    InferenceOutput output = runner.createOutputBuffer();
    fillInput(input, encodings, 0, arguments.batchSize);
    for (const auto iteration : range(WARMUP_ITERATIONS)) {
        static_cast<void>(iteration);
        runner.forwardInto(input, arguments.batchSize, output);
    }

    double checksum = 0.0;
    const auto startedAt = std::chrono::steady_clock::now();
    for (const auto iteration : range(arguments.iterations)) {
        static_cast<void>(iteration);
        runner.forwardInto(input, arguments.batchSize, output);
        const float *policyData = output.policies.data_ptr<float>();
        const float *outcomeData = output.outcomes.data_ptr<float>();
        for (const auto position : range(arguments.batchSize)) {
            const SearchInferenceResult<ChessGame> result = processInferencePosition<ChessGame>(
                policyData + position * ChessEncoding::action_count, outcomeData + position * 3,
                boards[position]);
            checksum += result.value() + static_cast<double>(result.actions.size());
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds}, {"checksum", checksum}};
}

nlohmann::json runPipeline(const Arguments &arguments, const std::vector<Board> &boards,
                           const std::vector<CompressedEncodedBoard> &encodings) {
    InferencePipeline pipeline(arguments.modelPath, arguments.device, 0, arguments.batchSize,
                               arguments.slots, true, ChessGame::Encoding::inferenceDimensions());
    std::vector<std::pair<size_t, size_t>> pendingBatches;
    pendingBatches.reserve(arguments.slots);
    constexpr size_t ENCODED_BOARD_BYTES = ChessRepresentationDimensions::channel_count *
                                           ChessRepresentationDimensions::board_length *
                                           ChessRepresentationDimensions::board_length;
    for (const auto warmupIteration : range(WARMUP_ITERATIONS)) {
        static_cast<void>(warmupIteration);
        const InferencePipeline::WritableBatch warmup = pipeline.acquireWritableBatch();
        for (const auto index : range(arguments.batchSize)) {
            encodings[index % encodings.size()].writeTensorInto(
                std::span(warmup.data + index * ENCODED_BOARD_BYTES, ENCODED_BOARD_BYTES));
        }
        pipeline.submit(warmup.slotIndex, arguments.batchSize);
        static_cast<void>(pipeline.consume<ChessGame>(
            warmup.slotIndex, std::span<const Board>(boards).first(arguments.batchSize)));
    }
    double checksum = 0.0;
    const auto startedAt = std::chrono::steady_clock::now();
    for (const auto iteration : range(arguments.iterations)) {
        if (pendingBatches.size() == arguments.slots) {
            const auto [slotIndex, firstBoard] = pendingBatches.front();
            const std::vector<SearchInferenceResult<ChessGame>> results =
                pipeline.consume<ChessGame>(slotIndex, std::span<const Board>(boards).subspan(
                                                           firstBoard, arguments.batchSize));
            for (const SearchInferenceResult<ChessGame> &result : results) {
                checksum += result.value() + static_cast<double>(result.actions.size());
            }
            pendingBatches.erase(pendingBatches.begin());
        }
        const InferencePipeline::WritableBatch writable = pipeline.acquireWritableBatch();
        for (const auto index : range(arguments.batchSize)) {
            encodings[(iteration * arguments.batchSize + index) % encodings.size()].writeTensorInto(
                std::span(writable.data + index * ENCODED_BOARD_BYTES, ENCODED_BOARD_BYTES));
        }
        pipeline.submit(writable.slotIndex, arguments.batchSize);
        pendingBatches.emplace_back(writable.slotIndex, iteration * arguments.batchSize);
    }
    for (const auto [slotIndex, firstBoard] : pendingBatches) {
        const std::vector<SearchInferenceResult<ChessGame>> results = pipeline.consume<ChessGame>(
            slotIndex, std::span<const Board>(boards).subspan(firstBoard, arguments.batchSize));
        for (const SearchInferenceResult<ChessGame> &result : results) {
            checksum += result.value() + static_cast<double>(result.actions.size());
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds}, {"checksum", checksum}};
}

nlohmann::json runReplicas(const Arguments &arguments,
                           const std::vector<CompressedEncodedBoard> &encodings) {
    std::vector<std::unique_ptr<InferenceRunner>> runners;
    std::vector<torch::Tensor> inputs;
    std::vector<InferenceOutput> outputs;
    runners.reserve(arguments.workers);
    inputs.reserve(arguments.workers);
    outputs.reserve(arguments.workers);
    for (const auto worker : range(arguments.workers)) {
        auto runner = std::make_unique<InferenceRunner>(arguments.modelPath, arguments.device, 0,
                                                        arguments.batchSize, true,
                                                        ChessGame::Encoding::inferenceDimensions());
        torch::Tensor input = runner->createInputBuffer();
        InferenceOutput output = runner->createOutputBuffer();
        fillInput(input, encodings, worker * arguments.batchSize, arguments.batchSize);
        for (const auto iteration : range(WARMUP_ITERATIONS)) {
            static_cast<void>(iteration);
            runner->forwardInto(input, arguments.batchSize, output);
        }
        inputs.push_back(std::move(input));
        outputs.push_back(std::move(output));
        runners.push_back(std::move(runner));
    }

    std::barrier startBarrier(static_cast<std::ptrdiff_t>(arguments.workers + 1));
    std::vector<std::thread> threads;
    threads.reserve(arguments.workers);
    for (const auto worker : range(arguments.workers)) {
        threads.emplace_back([&, worker]() {
            startBarrier.arrive_and_wait();
            for (const auto iteration : range(arguments.iterations)) {
                static_cast<void>(iteration);
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
    for (const auto worker : range(arguments.workers)) {
        auto runner = std::make_unique<InferenceRunner>(arguments.modelPath, arguments.device, 0,
                                                        arguments.batchSize, true,
                                                        ChessGame::Encoding::inferenceDimensions());
        torch::Tensor input = runner->createInputBuffer();
        InferenceOutput output = runner->createOutputBuffer();
        fillInput(input, encodings, worker * arguments.batchSize, arguments.batchSize);
        for (const auto iteration : range(WARMUP_ITERATIONS)) {
            static_cast<void>(iteration);
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
    for (const auto worker : range(arguments.workers)) {
        threads.emplace_back([&, worker]() {
            startBarrier.arrive_and_wait();
            for (const auto iteration : range(arguments.iterations)) {
                static_cast<void>(iteration);
                runners[worker]->forwardInto(inputs[worker], arguments.batchSize, outputs[worker]);
                const float *policyData = outputs[worker].policies.data_ptr<float>();
                const float *outcomeData = outputs[worker].outcomes.data_ptr<float>();
                for (const auto position : range(arguments.batchSize)) {
                    const size_t boardIndex = worker * arguments.batchSize + position;
                    const SearchInferenceResult<ChessGame> result =
                        processInferencePosition<ChessGame>(
                            policyData + position * ChessEncoding::action_count,
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
        configureSdpaBackend(arguments);
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
        for (const auto index : range(positions)) {
            encodings[index].writeTensorInto(std::span(
                packedEncodings.data() + index * ENCODED_BOARD_BYTES, ENCODED_BOARD_BYTES));
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
            result = runPipeline(arguments, boards, encodings);
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
        result["sdpa_backend"] = arguments.sdpaBackend;
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
