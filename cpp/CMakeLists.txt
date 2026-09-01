cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(AlphaZeroSelfPlay LANGUAGES CXX)

option(BUILD_TESTING "Build native C++ tests" OFF)
option(BUILD_BENCHMARKS "Build native C++ benchmarks" OFF)
include(CTest)

# --- Export compile commands for Clang-Tidy ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Optional: Enable clang-tidy integration ---
# Off by default: the naming check costs ~1 min per TU, which would dominate node provisioning builds.
# Enable for lint passes with -DENABLE_CLANG_TIDY=ON (or use cpp/run-clang-tidy.sh).
option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
if (ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if (CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        # Diagnostics only: -fix would rewrite sources (including fetched dependencies) during a build.
        set(CMAKE_CXX_CLANG_TIDY
                ${CLANG_TIDY_EXE}
                -checks=readability-identifier-naming
                -format-style=file
        )
    else ()
        message(WARNING "clang-tidy not found!")
    endif ()
endif ()

# Set default build type if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
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
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS
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

target_link_libraries(AlphaZeroCpp PRIVATE "${TORCH_LIBRARIES}")

if (BUILD_TESTING)
    add_executable(
            NativeTests
            test/TestMain.cpp
            test/TestBatchedSearch.cpp
            test/TestSearchStopping.cpp
            test/TestBitBoard.cpp
            test/TestBoardLegalMovesCache.cpp
            test/TestChessEncoding.cpp
            test/TestChessGame.cpp
            test/TestGameHistory.cpp
            test/TestGameSearchTree.cpp
            test/TestGoGame.cpp
            test/TestInferencePipeline.cpp
            test/TestMovePolicyProcessing.cpp
            test/TestPackedPlaneFixtures.cpp
            src/games/chess/implementation/ChessAction.cpp
            src/games/chess/implementation/ChessBoard.cpp
            src/games/chess/encoding/ChessEncoding.cpp
            src/games/chess/encoding/ChessPolicyEncoding.cpp
            src/games/chess/presentation/ChessSearchPresentation.cpp
            src/search/InferencePipeline.cpp
            src/search/SearchStopPredictor.cpp
            "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
            "${stockfish_SOURCE_DIR}/src/memory.cpp"
            "${stockfish_SOURCE_DIR}/src/misc.cpp"
            "${stockfish_SOURCE_DIR}/src/movegen.cpp"
            "${stockfish_SOURCE_DIR}/src/position.cpp"
    )
    target_include_directories(NativeTests PRIVATE src test)
    target_compile_definitions(NativeTests PRIVATE SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(NativeTests PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
    add_test(NAME NativeTests COMMAND NativeTests)
endif ()

if (BUILD_BENCHMARKS)
add_executable(
        InferenceBenchmark
        benchmark/InferenceBenchmark.cpp
        src/games/chess/implementation/ChessBoard.cpp
        src/games/chess/encoding/ChessEncoding.cpp
        src/games/chess/encoding/ChessPolicyEncoding.cpp
        src/search/InferencePipeline.cpp
        src/search/SearchStopPredictor.cpp
        src/games/chess/implementation/ChessAction.cpp
        "${stockfish_SOURCE_DIR}/src/bitboard.cpp"
        "${stockfish_SOURCE_DIR}/src/memory.cpp"
        "${stockfish_SOURCE_DIR}/src/misc.cpp"
        "${stockfish_SOURCE_DIR}/src/movegen.cpp"
        "${stockfish_SOURCE_DIR}/src/position.cpp"
)
target_include_directories(InferenceBenchmark PRIVATE src)
target_link_libraries(InferenceBenchmark PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
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
    # The stub is restyled after generation, so a rebuild leaves the checkout clean and run_control can start.
    execute_process(
            COMMAND ${Python3_EXECUTABLE} -m ruff --version
            RESULT_VARIABLE ruff_probe_result
            OUTPUT_QUIET
            ERROR_QUIET
    )
    if (NOT ruff_probe_result EQUAL 0)
        message(FATAL_ERROR
                "ruff is unavailable to ${Python3_EXECUTABLE}; the generated stub cannot be restyled. "
                "Install the dev dependency group (uv sync) or configure with -DCMAKE_BUILD_TYPE=CompileCheck.")
    endif ()

    add_custom_command(TARGET AlphaZeroCpp POST_BUILD
            COMMENT "Copying AlphaZeroCpp.so and generating stubs..."

            # Make sure the output Python directory exists
            COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_MODULE_OUTPUT_DIR}

            # Copy the compiled .so to the Python project directory
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:AlphaZeroCpp> ${PYTHON_MODULE_OUTPUT_DIR}

            # Generate the stub with correct PYTHONPATH
            COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${PYTHON_MODULE_OUTPUT_DIR}
            ${Python3_EXECUTABLE} -m pybind11_stubgen AlphaZeroCpp -o ${CMAKE_CURRENT_BINARY_DIR}/stubs

            # The committed stub IS the generated stub: hand edits drift from the bindings.
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/stubs/AlphaZeroCpp.pyi ${PYTHON_MODULE_OUTPUT_DIR}/AlphaZeroCpp.pyi

            # Restyle in place so regeneration is a no-op against the committed copy.
            COMMAND ${Python3_EXECUTABLE} -m ruff check --fix --quiet
            ${PYTHON_MODULE_OUTPUT_DIR}/AlphaZeroCpp.pyi
            COMMAND ${Python3_EXECUTABLE} -m ruff format --quiet
            ${PYTHON_MODULE_OUTPUT_DIR}/AlphaZeroCpp.pyi
    )
endif ()


# --- Export compile commands ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
