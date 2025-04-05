cmake_minimum_required(VERSION 3.23.3 FATAL_ERROR)
project(AlphaZeroSelfPlay LANGUAGES CXX)

# Enable position-independent code for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Check for CUDA support ---
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
endif()

# --- Set C++ and CUDA standards to C++20---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif()

# --- Enable ccache for faster recompilation ---
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Using ccache")
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    if(CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
    endif()
endif()

# --- LibTorch configuration ---
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cuda")
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cuda/share/cmake/Torch")
elseif(UNIX)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux")
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux/share/cmake/Torch")
elseif(MSVC)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_win")
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_win/share/cmake/Torch")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()
find_package(Torch REQUIRED)

# --- Add tensorboard_logger via FetchContent ---
find_package(Protobuf REQUIRED)
include(FetchContent)
FetchContent_Declare(
    tensorboard_logger
    GIT_REPOSITORY https://github.com/RustingSword/tensorboard_logger.git
    GIT_TAG master
)
FetchContent_MakeAvailable(tensorboard_logger)

# --- Add fmtlib via FetchContent ---
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 8.1.1  # Or another version tag
)
FetchContent_MakeAvailable(fmt)

# --- FetchContent for pybind11 ---
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6  # Use the desired version
)
FetchContent_MakeAvailable(pybind11)

# --- Include directories ---
include_directories(src)
if(CMAKE_CUDA_COMPILER)
    include_directories(libtorch_cuda/include)
elseif(UNIX)
    include_directories(libtorch_cpu_linux/include)
elseif(MSVC)
    include_directories(libtorch_cpu_win/include)
endif()

# --- Collect sources for the Python module ---
# Gather all .cpp files under src/
file(GLOB_RECURSE MODULE_SOURCES "src/*.cpp")

# Precompiled headers for src/common.hpp

set(PCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/common.hpp")


# --- Build the Python extension module ---
pybind11_add_module(AlphaZeroCpp ${MODULE_SOURCES})

# --- Set compile options (optional) ---
if(MSVC)
    set(debug_flags_list "/Od" "/Zi")
    set(release_flags_list "/O2" "/DNDEBUG")
else()
    set(debug_flags_list "-Og" "-g")
    set(release_flags_list "-O3" "-ffast-math" "-DNDEBUG" "-march=native")
endif()

target_compile_options(AlphaZeroCpp PRIVATE
    $<$<CONFIG:DEBUG>:${debug_flags_list}>
    $<$<CONFIG:RELEASE>:${release_flags_list}>
)
if(CMAKE_CUDA_COMPILER)
    if(MSVC)
        set(cuda_debug_flags_list "-G" "-g")
        set(cuda_release_flags_list "-O3" "--use_fast_math" "-DNDEBUG")
    else()
        set(cuda_debug_flags_list "-O0" "-G" "-g")
        set(cuda_release_flags_list "-O3" "--use_fast_math" "-DNDEBUG")
    endif()
    target_compile_options(AlphaZeroCpp PRIVATE
        $<$<CONFIG:DEBUG>:${cuda_debug_flags_list}>
        $<$<CONFIG:RELEASE>:${cuda_release_flags_list}>
    )
endif()

# --- Link libraries ---
target_link_libraries(AlphaZeroCpp PRIVATE "${TORCH_LIBRARIES}" tensorboard_logger fmt::fmt)
if(MSVC)
    target_link_options(AlphaZeroCpp PRIVATE $<$<CONFIG:RELEASE>:"/LTCG">)
endif()
if(UNIX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(AlphaZeroCpp stdc++fs)
endif()

# --- Export compile commands ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Post-build: Copy the generated shared library to "../../cpp_py/" ---
add_custom_command(TARGET AlphaZeroCpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/../../cpp_py/"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:AlphaZeroCpp> "${CMAKE_CURRENT_BINARY_DIR}/../../cpp_py/"
    COMMENT "Copying AlphaZeroCpp module to ../../cpp_py/"
)
