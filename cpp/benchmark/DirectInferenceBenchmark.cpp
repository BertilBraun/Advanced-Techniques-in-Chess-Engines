#include "Board.h"
#include "BoardEncoding.hpp"
#include "DirectInference.hpp"
#include "InferenceClient.hpp"
#include "NonCachingInferenceClient.hpp"

#include <barrier>
#include <nlohmann/json.hpp>

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
        const std::vector<Move> &moves = board.validMoves();
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
        encodings.push_back(encodeBoard(&board));
    }
    const auto finishedAt = std::chrono::steady_clock::now();
    elapsedMilliseconds = std::chrono::duration<double, std::milli>(finishedAt - startedAt).count();
    return encodings;
}

void fillInput(torch::Tensor &input, const std::vector<CompressedEncodedBoard> &encodings,
               const size_t firstEncoding, const size_t batchSize) {
    int8 *destination = input.data_ptr<int8>();
    constexpr size_t ENCODED_BOARD_BYTES = BOARD_C * BOARD_LEN * BOARD_LEN;
    for (size_t index = 0; index < batchSize; ++index) {
        writeTensorEncoding(encodings[(firstEncoding + index) % encodings.size()],
                            destination + index * ENCODED_BOARD_BYTES);
    }
}

double outputChecksum(const DirectInferenceOutput &output) {
    return output.policies.sum().item<double>() + output.outcomes.sum().item<double>();
}

nlohmann::json runDirect(const Arguments &arguments,
                         const std::vector<CompressedEncodedBoard> &encodings) {
    DirectInferenceRunner runner(arguments.modelPath, arguments.device, 0, arguments.batchSize,
                                 true);
    torch::Tensor input = runner.createInputBuffer();
    DirectInferenceOutput output = runner.createOutputBuffer();
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

nlohmann::json runPipeline(const Arguments &arguments,
                           const std::vector<CompressedEncodedBoard> &encodings) {
    DirectInferencePipeline pipeline(arguments.modelPath, arguments.device, 0, arguments.batchSize,
                                     arguments.slots, true);
    std::vector<size_t> pendingSlots;
    pendingSlots.reserve(arguments.slots);
    constexpr size_t ENCODED_BOARD_BYTES = BOARD_C * BOARD_LEN * BOARD_LEN;
    for (size_t warmupIteration = 0; warmupIteration < WARMUP_ITERATIONS; ++warmupIteration) {
        const DirectInferencePipeline::WritableBatch warmup = pipeline.acquireWritableBatch();
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
        const DirectInferencePipeline::WritableBatch writable = pipeline.acquireWritableBatch();
        for (size_t index = 0; index < arguments.batchSize; ++index) {
            writeTensorEncoding(
                encodings[(iteration * arguments.batchSize + index) % encodings.size()],
                writable.data + index * ENCODED_BOARD_BYTES);
        }
        pipeline.submit(writable.slotIndex, arguments.batchSize);
        pendingSlots.push_back(writable.slotIndex);
    }
    DirectInferenceOutput finalOutput;
    for (const size_t slotIndex : pendingSlots) {
        const DirectInferenceOutput output = pipeline.waitCompleted(slotIndex);
        finalOutput = output;
        pipeline.release(slotIndex);
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds}, {"checksum", outputChecksum(finalOutput)}};
}

nlohmann::json runReplicas(const Arguments &arguments,
                           const std::vector<CompressedEncodedBoard> &encodings) {
    std::vector<std::unique_ptr<DirectInferenceRunner>> runners;
    std::vector<torch::Tensor> inputs;
    std::vector<DirectInferenceOutput> outputs;
    runners.reserve(arguments.workers);
    inputs.reserve(arguments.workers);
    outputs.reserve(arguments.workers);
    for (size_t worker = 0; worker < arguments.workers; ++worker) {
        auto runner = std::make_unique<DirectInferenceRunner>(arguments.modelPath, arguments.device,
                                                              0, arguments.batchSize, true);
        torch::Tensor input = runner->createInputBuffer();
        DirectInferenceOutput output = runner->createOutputBuffer();
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
    for (const DirectInferenceOutput &output : outputs) {
        checksum += outputChecksum(output);
    }
    return {{"elapsed_seconds", seconds}, {"checksum", checksum}};
}

template <typename Client>
nlohmann::json runClient(const Arguments &arguments, const std::vector<Board> &boards,
                         const size_t cacheCapacity) {
    const InferenceClientParams parameters(0, arguments.modelPath,
                                           static_cast<int>(arguments.batchSize), 0, cacheCapacity,
                                           arguments.device);
    Client client(parameters);
    double checksum = 0.0;
    std::vector<const Board *> warmupBatch;
    warmupBatch.reserve(arguments.batchSize);
    for (size_t index = 0; index < arguments.batchSize; ++index) {
        warmupBatch.push_back(&boards[index]);
    }
    for (size_t iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        static_cast<void>(client.inferenceBatch(warmupBatch));
    }
    const auto startedAt = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < arguments.iterations; ++iteration) {
        std::vector<const Board *> batch;
        batch.reserve(arguments.batchSize);
        for (size_t index = 0; index < arguments.batchSize; ++index) {
            batch.push_back(&boards[(iteration * arguments.batchSize + index) % boards.size()]);
        }
        const std::vector<InferenceResult> results = client.inferenceBatch(batch);
        for (const InferenceResult &result : results) {
            checksum += result.outcome.expectedValue();
        }
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {{"elapsed_seconds", seconds}, {"checksum", checksum}};
}
} // namespace

int main(const int argumentCount, char **argumentValues) {
    try {
        Bitboards::init();
        Position::init();
        const Arguments arguments = parseArguments(argumentCount, argumentValues);
        const size_t positions = arguments.batchSize * arguments.iterations * arguments.workers;
        const std::vector<Board> boards = generateBoards(positions, arguments.seed);
        double encodeMilliseconds = 0.0;
        const std::vector<CompressedEncodedBoard> encodings =
            encodeBoards(boards, encodeMilliseconds);
        constexpr size_t ENCODED_BOARD_BYTES = BOARD_C * BOARD_LEN * BOARD_LEN;
        std::vector<int8> packedEncodings(positions * ENCODED_BOARD_BYTES);
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
        } else if (arguments.mode == "pipeline") {
            result = runPipeline(arguments, encodings);
        } else if (arguments.mode == "replicas") {
            result = runReplicas(arguments, encodings);
        } else if (arguments.mode == "cached") {
            result = runClient<InferenceClient>(arguments, boards, positions * 2);
        } else if (arguments.mode == "noncached") {
            result = runClient<NonCachingInferenceClient>(arguments, boards, 0);
        } else {
            throw std::invalid_argument(
                "Mode must be direct, pipeline, replicas, cached, or noncached");
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
