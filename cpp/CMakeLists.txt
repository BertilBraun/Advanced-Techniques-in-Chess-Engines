cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(AlphaZeroSelfPlay LANGUAGES CXX)

option(BUILD_TESTING "Build native C++ tests" OFF)
option(BUILD_BENCHMARKS "Build native C++ benchmarks" OFF)
include(CTest)

# --- Export compile commands for Clang-Tidy ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Optional: Enable clang-tidy integration ---
option(ENABLE_CLANG_TIDY "Enable clang-tidy" ON)
if (ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if (CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        # TODO set(CMAKE_CXX_CLANG_TIDY
        # TODO     ${CLANG_TIDY_EXE}
        # TODO     -checks=readability-identifier-naming
        # TODO     -fix
        # TODO     -format-style=file
        # TODO     -warnings-as-errors=*
        # TODO )
    else ()
        message(WARNING "clang-tidy not found!")
    endif ()
endif ()

# Set default build type if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
option(ENABLE_TIMING_INSTRUMENTATION "Enable TIMEIT instrumentation" OFF)
message(STATUS "Timing instrumentation: ${ENABLE_TIMING_INSTRUMENTATION}")
option(ENABLE_NATIVE_ARCHITECTURE "Optimize for the build machine CPU" ON)
message(STATUS "Native architecture optimization: ${ENABLE_NATIVE_ARCHITECTURE}")

# Add debug flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Adding debug flags (-g -O3)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -fsanitize=address -fsanitize=undefined -D_GLIBCXX_ASSERTIONS -fno-omit-frame-pointer")
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O3")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Adding release flags (-O3 -DNDEBUG)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -flto")
    if (ENABLE_NATIVE_ARCHITECTURE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif ()
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Adding release with debug info flags (-O2 -g)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -fno-omit-frame-pointer")
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -G")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "CompileCheck")
    message(STATUS "Adding compile-check flags (-O0 -g0)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g0")
else ()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif ()

# Enable position-independent code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Check and enable CUDA (if available) ---
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
endif ()

# --- Set C++ and CUDA standards to C++20 ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif ()

# --- Enable ccache for faster recompilation (if available) ---
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    message(STATUS "Using ccache")
    set(CCACHE_LAUNCHER
            ${CMAKE_COMMAND} -E env
            CCACHE_SLOPPINESS=pch_defines,time_macros
            ${CCACHE_FOUND}
    )
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_LAUNCHER})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_LAUNCHER})
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fpch-preprocess)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fno-pch-timestamp)
    endif ()
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_LAUNCHER})
    endif ()
endif ()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# --- Dynamically set CMAKE_PREFIX_PATH using Python and torch ---
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to discover the Torch CMake prefix with ${Python3_EXECUTABLE}")
endif ()
message(STATUS "Setting CMAKE_PREFIX_PATH to: ${TORCH_CMAKE_PREFIX_PATH}")
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)

# --- Dynamically set the GLIBCXX flag using Python and torch ---
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
        OUTPUT_VARIABLE TORCH_GLIBCXX_FLAG
        RESULT_VARIABLE TORCH_GLIBCXX_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT TORCH_GLIBCXX_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to discover the Torch C++ ABI with ${Python3_EXECUTABLE}")
endif ()
message(STATUS "Torch reports _GLIBCXX_USE_CXX11_ABI as: ${TORCH_GLIBCXX_FLAG}")
if (TORCH_GLIBCXX_FLAG STREQUAL "True")
    set(GLIBCXX_ABI_VAL 1)
else ()
    set(GLIBCXX_ABI_VAL 0)
endif ()
message(STATUS "Setting _GLIBCXX_USE_CXX11_ABI to: ${GLIBCXX_ABI_VAL}")

# Append LibTorch compiler flags and ABI flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI_VAL}")
if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CUDA_FLAGS}")
endif ()

# --- Fetch pybind11 for Python bindings ---
include(FetchContent)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "No dev warnings")
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4
)

# --- Fetch Stockfish ---
FetchContent_Declare(
        Stockfish
        GIT_REPOSITORY https://github.com/BertilBraun/Stockfish.git
        GIT_TAG 22956373a5618da5634966e22a250a0482680d9c
        PATCH_COMMAND
        ${CMAKE_COMMAND}
        -DREPOSITORY=<SOURCE_DIR>
        -DPATCH=${CMAKE_CURRENT_SOURCE_DIR}/patches/stockfish-bounded-repetition.patch
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ApplyPatchOnce.cmake
)

# --- Fetch Nlohmann JSON ---
FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG 55f93686c01528224f448c19128836e7df245f72
)

FetchContent_MakeAvailable(pybind11 Stockfish nlohmann_json) # tensorboard_logger)


# find_package(Protobuf REQUIRED)

# --- Include directories ---
include_directories(src)
# include Stockfish include directory
include_directories(${stockfish_SOURCE_DIR}/src)
include_directories(${nlohmann_json_SOURCE_DIR}/single_include/nlohmann)

# --- Collect all source files from src/ and Stockfish/src/ ---
file(GLOB_RECURSE ALL_SOURCES
        "src/*.cpp"
        "src/*.h"
        "src/*.hpp"
        "${stockfish_SOURCE_DIR}/src/*.cpp"
        "${stockfish_SOURCE_DIR}/src/*.h"
)

# Build the Python module from the binding entry point and shared sources
if (CMAKE_BUILD_TYPE STREQUAL "CompileCheck")
    pybind11_add_module(AlphaZeroCpp NO_EXTRAS ${ALL_SOURCES})
else ()
    pybind11_add_module(AlphaZeroCpp ${ALL_SOURCES})
endif ()

target_precompile_headers(AlphaZeroCpp PRIVATE src/common.hpp)
if (ENABLE_TIMING_INSTRUMENTATION)
    target_compile_definitions(AlphaZeroCpp PRIVATE ENABLE_TIMING)
endif ()

target_link_libraries(AlphaZeroCpp PRIVATE "${TORCH_LIBRARIES}")

if (BUILD_TESTING)
add_executable(BlockingQueueTest test/TestBlockingQueue.cpp)
target_include_directories(BlockingQueueTest PRIVATE src)
add_test(NAME BlockingQueueTest COMMAND BlockingQueueTest)

add_executable(BitBoardTest test/TestBitBoard.cpp)
target_include_directories(BitBoardTest PRIVATE src)
add_test(NAME BitBoardTest COMMAND BitBoardTest)

add_executable(
        PackedPlaneFixtureTest
        test/TestPackedPlaneFixtures.cpp
        src/games/chess/ChessBoard.cpp
        src/games/chess/ChessEncoding.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(PackedPlaneFixtureTest PRIVATE src)
target_compile_definitions(PackedPlaneFixtureTest PRIVATE SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(PackedPlaneFixtureTest PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
add_test(NAME PackedPlaneFixtureTest COMMAND PackedPlaneFixtureTest)

add_executable(
        BoardLegalMovesCacheTest
        test/TestBoardLegalMovesCache.cpp
        src/games/chess/ChessBoard.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(BoardLegalMovesCacheTest PRIVATE src)
target_link_libraries(BoardLegalMovesCacheTest PRIVATE "${TORCH_LIBRARIES}")
add_test(NAME BoardLegalMovesCacheTest COMMAND BoardLegalMovesCacheTest)

add_executable(
        ChessGameContractTest
        test/TestChessGameContract.cpp
        src/games/chess/ChessBoard.cpp
        src/games/chess/ChessEncoding.cpp
        src/games/chess/ChessHistory.cpp
        src/games/chess/ChessAction.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(ChessGameContractTest PRIVATE src)
target_link_libraries(ChessGameContractTest PRIVATE "${TORCH_LIBRARIES}")
add_test(NAME ChessGameContractTest COMMAND ChessGameContractTest)

add_executable(GoGameContractTest test/TestGoGameContract.cpp)
target_include_directories(GoGameContractTest PRIVATE src)
add_test(NAME GoGameContractTest COMMAND GoGameContractTest)

add_executable(
        GameHistoryTest
        test/TestGameHistory.cpp
        src/games/chess/ChessBoard.cpp
        src/games/chess/ChessHistory.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(GameHistoryTest PRIVATE src)
target_link_libraries(GameHistoryTest PRIVATE "${TORCH_LIBRARIES}")
add_test(NAME GameHistoryTest COMMAND GameHistoryTest)

add_executable(
        GameSearchTreeTest
        test/TestGameSearchTree.cpp
        src/games/chess/ChessBoard.cpp
        src/games/chess/ChessEncoding.cpp
        src/games/chess/ChessHistory.cpp
        src/games/chess/ChessAction.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(GameSearchTreeTest PRIVATE src)
target_link_libraries(GameSearchTreeTest PRIVATE "${TORCH_LIBRARIES}")
add_test(NAME GameSearchTreeTest COMMAND GameSearchTreeTest)

add_executable(
        MovePolicyProcessingTest
        test/TestMovePolicyProcessing.cpp
        src/games/chess/ChessBoard.cpp
        src/games/chess/ChessAction.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(MovePolicyProcessingTest PRIVATE src)
target_link_libraries(MovePolicyProcessingTest PRIVATE "${TORCH_LIBRARIES}")
add_test(NAME MovePolicyProcessingTest COMMAND MovePolicyProcessingTest)

add_executable(
        DirectBoardEncodingTest
        test/TestDirectBoardEncoding.cpp
        src/games/chess/ChessBoard.cpp
        src/games/chess/ChessEncoding.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(DirectBoardEncodingTest PRIVATE src)
target_link_libraries(DirectBoardEncodingTest PRIVATE "${TORCH_LIBRARIES}")
add_test(NAME DirectBoardEncodingTest COMMAND DirectBoardEncodingTest)

add_executable(
        DirectInferencePipelineTest
        test/TestDirectInferencePipeline.cpp
        src/DirectInference.cpp
)
target_include_directories(DirectInferencePipelineTest PRIVATE src)
target_link_libraries(DirectInferencePipelineTest PRIVATE "${TORCH_LIBRARIES}")
add_test(NAME DirectInferencePipelineTest COMMAND DirectInferencePipelineTest)

add_executable(
        BatchedSearchTest
        test/TestBatchedSearch.cpp
        src/games/chess/ChessBoard.cpp
        src/games/chess/ChessEncoding.cpp
        src/DirectInference.cpp
        src/MCTS/MCTS.cpp
        src/games/chess/ChessSearch.cpp
        src/games/chess/ChessAction.cpp
        src/util/Log.cpp
        src/util/ThreadPool.cpp
        src/util/TimeItGuard.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(BatchedSearchTest PRIVATE src)
target_link_libraries(BatchedSearchTest PRIVATE "${TORCH_LIBRARIES}")
add_test(NAME BatchedSearchTest COMMAND BatchedSearchTest)
endif ()

if (BUILD_BENCHMARKS)
add_executable(
        DirectInferenceBenchmark
        benchmark/DirectInferenceBenchmark.cpp
        src/games/chess/ChessBoard.cpp
        src/games/chess/ChessEncoding.cpp
        src/DirectInference.cpp
        src/games/chess/ChessAction.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(DirectInferenceBenchmark PRIVATE src)
target_link_libraries(DirectInferenceBenchmark PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
endif ()

set_target_properties(AlphaZeroCpp PROPERTIES
        CXX_STANDARD 20
        PREFIX ""
        SUFFIX ".so"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)

# Post-build step to copy the module and generate the stubs
set(PYTHON_MODULE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../py")

if (NOT CMAKE_BUILD_TYPE STREQUAL "CompileCheck")
    add_custom_command(TARGET AlphaZeroCpp POST_BUILD
            COMMENT "Copying AlphaZeroCpp.so and generating stubs..."

            # Make sure the output Python directory exists
            COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_MODULE_OUTPUT_DIR}

            # Copy the compiled .so to the Python project directory
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:AlphaZeroCpp> ${PYTHON_MODULE_OUTPUT_DIR}

            # Generate the stub with correct PYTHONPATH
            COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${PYTHON_MODULE_OUTPUT_DIR}
            pybind11-stubgen AlphaZeroCpp -o ${CMAKE_CURRENT_BINARY_DIR}/stubs
    )
endif ()


# --- Export compile commands ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
