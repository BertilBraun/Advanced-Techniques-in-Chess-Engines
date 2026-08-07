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
    NativeTestSuite{"BatchedSearch", runBatchedSearchTests},
    NativeTestSuite{"BitBoard", runBitBoardTests},
    NativeTestSuite{"BlockingQueue", runBlockingQueueTests},
    NativeTestSuite{"BoardLegalMovesCache", runBoardLegalMovesCacheTests},
    NativeTestSuite{"ChessEncoding", runChessEncodingTests},
    NativeTestSuite{"ChessGameContract", runChessGameContractTests},
    NativeTestSuite{"GameHistory", runGameHistoryTests},
    NativeTestSuite{"GameSearchTree", runGameSearchTreeTests},
    NativeTestSuite{"GoGameContract", runGoGameContractTests},
    NativeTestSuite{"InferencePipeline", runInferencePipelineTests},
    NativeTestSuite{"MovePolicyProcessing", runMovePolicyProcessingTests},
    NativeTestSuite{"PackedPlaneFixture", runPackedPlaneFixtureTests},
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
