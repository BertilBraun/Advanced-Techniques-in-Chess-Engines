cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(AlphaZeroGo LANGUAGES CXX)

include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(AZ_BUILD_PYTHON "Build the az_go_native Python extension" OFF)
option(ENABLE_NATIVE_ARCHITECTURE "Optimize for the build machine CPU" ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

function(az_enable_warnings target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    else ()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif ()
endfunction()

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
FetchContent_MakeAvailable(Stockfish)

add_library(
        az_stockfish_rules
        ${stockfish_SOURCE_DIR}/src/bitboard.cpp
        ${stockfish_SOURCE_DIR}/src/memory.cpp
        ${stockfish_SOURCE_DIR}/src/misc.cpp
        ${stockfish_SOURCE_DIR}/src/movegen.cpp
        ${stockfish_SOURCE_DIR}/src/position.cpp
)
target_include_directories(az_stockfish_rules SYSTEM PUBLIC ${stockfish_SOURCE_DIR}/src)

add_library(az_core INTERFACE)
target_include_directories(az_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_sources(
        az_core
        INTERFACE
        src/core/ArtifactMetadata.hpp
        src/core/SessionConfiguration.hpp
        src/games/GameDefinition.hpp
        src/games/game_concepts.hpp
)

add_library(az_search INTERFACE)
target_link_libraries(az_search INTERFACE az_core)
target_sources(
        az_search
        INTERFACE
        src/inference/InferenceBatch.hpp
        src/inference/InferenceTypes.hpp
        src/search/FixedPuct.hpp
        src/search/SearchConfiguration.hpp
        src/search/SearchTelemetry.hpp
        src/search/SeededRandom.hpp
        src/search/TreeArena.hpp
)

add_library(
        az_go
        src/games/go/GoDefinition.cpp
        src/games/go/GoEncoding.cpp
        src/games/go/GoState.cpp
        src/games/go/GoSymmetry.cpp
)
target_link_libraries(az_go PUBLIC az_core)
target_precompile_headers(az_go PRIVATE src/common.hpp)
az_enable_warnings(az_go)

add_library(
        az_chess
        src/games/chess/ChessDefinition.cpp
        src/games/chess/ChessEncoding.cpp
        src/games/chess/ChessPolicy.cpp
        src/games/chess/ChessState.cpp
)
target_link_libraries(az_chess PUBLIC az_core az_stockfish_rules)
target_precompile_headers(az_chess PRIVATE src/common.hpp)
az_enable_warnings(az_chess)

if (ENABLE_NATIVE_ARCHITECTURE AND NOT MSVC)
    target_compile_options(az_go PRIVATE -march=native)
    target_compile_options(az_chess PRIVATE -march=native)
endif ()

if (BUILD_TESTING)
    add_executable(GoStateTest test/TestGoState.cpp)
    target_link_libraries(GoStateTest PRIVATE az_go)
    target_precompile_headers(GoStateTest PRIVATE src/common.hpp)
    az_enable_warnings(GoStateTest)
    add_test(NAME GoStateTest COMMAND GoStateTest)

    add_executable(FixedPuctTest test/TestFixedPuct.cpp)
    target_link_libraries(FixedPuctTest PRIVATE az_go az_search)
    target_precompile_headers(FixedPuctTest PRIVATE src/common.hpp)
    az_enable_warnings(FixedPuctTest)
    add_test(NAME FixedPuctTest COMMAND FixedPuctTest)

    add_executable(GameContractsTest test/TestGameContracts.cpp)
    target_link_libraries(GameContractsTest PRIVATE az_go)
    target_precompile_headers(GameContractsTest PRIVATE src/common.hpp)
    az_enable_warnings(GameContractsTest)
    add_test(NAME GameContractsTest COMMAND GameContractsTest)

    add_executable(TreeArenaTest test/TestTreeArena.cpp)
    target_link_libraries(TreeArenaTest PRIVATE az_search)
    target_precompile_headers(TreeArenaTest PRIVATE src/common.hpp)
    az_enable_warnings(TreeArenaTest)
    add_test(NAME TreeArenaTest COMMAND TreeArenaTest)

    add_executable(ChessStateTest test/TestChessState.cpp)
    target_link_libraries(ChessStateTest PRIVATE az_chess)
    target_precompile_headers(ChessStateTest PRIVATE src/common.hpp)
    az_enable_warnings(ChessStateTest)
    add_test(NAME ChessStateTest COMMAND ChessStateTest)
endif ()

if (AZ_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(
            az_go_native
            src/bindings/module.cpp
            src/bindings/go_bindings.cpp
            src/bindings/search_bindings.cpp
    )
    target_link_libraries(az_go_native PRIVATE az_go az_search)
    target_precompile_headers(az_go_native PRIVATE src/common.hpp)
    az_enable_warnings(az_go_native)
endif ()
