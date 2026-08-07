#include "TestRunner.hpp"

#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string_view>

namespace {
struct NativeTestSuite {
    std::string_view name;
    int (*run)();
};

constexpr std::array testSuites = {
    NativeTestSuite{.name = "BatchedSearch", .run = runBatchedSearchTests},
    NativeTestSuite{.name = "BitBoard", .run = runBitBoardTests},
    NativeTestSuite{.name = "BoardLegalMovesCache", .run = runBoardLegalMovesCacheTests},
    NativeTestSuite{.name = "ChessEncoding", .run = runChessEncodingTests},
    NativeTestSuite{.name = "ChessGameContract", .run = runChessGameContractTests},
    NativeTestSuite{.name = "GameHistory", .run = runGameHistoryTests},
    NativeTestSuite{.name = "GameSearchTree", .run = runGameSearchTreeTests},
    NativeTestSuite{.name = "GoGameContract", .run = runGoGameContractTests},
    NativeTestSuite{.name = "InferencePipeline", .run = runInferencePipelineTests},
    NativeTestSuite{.name = "MovePolicyProcessing", .run = runMovePolicyProcessingTests},
    NativeTestSuite{.name = "PackedPlaneFixture", .run = runPackedPlaneFixtureTests},
};
} // namespace

int main() {
    std::size_t failedSuiteCount = 0;
    for (const NativeTestSuite &testSuite : testSuites) {
        std::cout << "[ RUN      ] " << testSuite.name << '\n';
        try {
            const int exitCode = testSuite.run();
            if (exitCode == EXIT_SUCCESS) {
                std::cout << "[       OK ] " << testSuite.name << '\n';
            } else {
                ++failedSuiteCount;
                std::cerr << "[  FAILED  ] " << testSuite.name << " (exit code " << exitCode
                          << ")\n";
            }
        } catch (const std::exception &exception) {
            ++failedSuiteCount;
            std::cerr << "[  FAILED  ] " << testSuite.name << ": " << exception.what() << '\n';
        } catch (...) {
            ++failedSuiteCount;
            std::cerr << "[  FAILED  ] " << testSuite.name << ": unknown exception\n";
        }
    }

    const std::size_t passedSuiteCount = testSuites.size() - failedSuiteCount;
    std::cout << "[==========] " << passedSuiteCount << " passed, " << failedSuiteCount
              << " failed\n";
    return failedSuiteCount == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
