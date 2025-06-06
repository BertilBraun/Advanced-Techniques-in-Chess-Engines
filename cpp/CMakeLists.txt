cmake_minimum_required(VERSION 3.23.3 FATAL_ERROR)
project(AlphaZeroSelfPlay LANGUAGES CXX)

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
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add debug flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Adding debug flags (-g -O0)")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Adding release flags (-O3)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Adding release with debug info flags (-O2 -g)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -fno-omit-frame-pointer")
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -G")
    endif ()
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
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif ()

# --- Enable ccache for faster recompilation (if available) ---
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    message(STATUS "Using ccache")
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    if (CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
    endif ()
endif ()

# --- Dynamically set CMAKE_PREFIX_PATH using Python and torch ---
execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Setting CMAKE_PREFIX_PATH to: ${TORCH_CMAKE_PREFIX_PATH}")
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)

# --- Dynamically set the GLIBCXX flag using Python and torch ---
execute_process(
        COMMAND python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
        OUTPUT_VARIABLE TORCH_GLIBCXX_FLAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
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
        GIT_TAG v2.13.6
)

# --- Fetch tensorboard_logger ---
# FetchContent_Declare(
#     tensorboard_logger
#     GIT_REPOSITORY https://github.com/RustingSword/tensorboard_logger.git
#     GIT_TAG master
# )

# --- Fetch Stockfish ---
FetchContent_Declare(
        Stockfish
        GIT_REPOSITORY https://github.com/official-stockfish/Stockfish.git
        GIT_TAG master
)

# --- Fetch Nlohmann JSON ---
FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG master
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

# Filter out NNUE files
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*/nnue/.*\\.cpp$")

# Build the Python module from the binding entry point and shared sources
pybind11_add_module(AlphaZeroCpp ${ALL_SOURCES})

target_precompile_headers(AlphaZeroCpp PRIVATE src/common.hpp)

target_link_libraries(AlphaZeroCpp PRIVATE "${TORCH_LIBRARIES}")

# target_link_libraries(AlphaZeroCpp PRIVATE ${Protobuf_LIBRARIES} tensorboard_logger "${TORCH_LIBRARIES}")

# target_include_directories(AlphaZeroCpp PRIVATE ${Protobuf_INCLUDE_DIRS})
# target_include_directories(AlphaZeroCpp PRIVATE ${tensorboard_logger_SOURCE_DIR}/include)

set_target_properties(AlphaZeroCpp PROPERTIES
        CXX_STANDARD 20
        PREFIX ""
        SUFFIX ".so"
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
)

# Optional: Post-build step to copy the module
add_custom_command(TARGET AlphaZeroCpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/../../cpp_py/"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:AlphaZeroCpp> "${CMAKE_CURRENT_BINARY_DIR}/../../cpp_py/"
        COMMENT "Copying AlphaZeroCpp module to ../../cpp_py/"
)

# --- Export compile commands ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
