cmake_minimum_required(VERSION 3.23.3 FATAL_ERROR)
project(AIZeroChessBot LANGUAGES CXX)

# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
endif()

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif()

# Enable ccache for faster recompilation
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Using ccache")
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    if(CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
    endif()
endif()

# LibTorch configuration
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

# Collect all source files (adjust pattern as needed)
file(GLOB_RECURSE SOURCES "src/*.cpp")

# General warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Print MSVC and UNIX values
message(STATUS "MSVC: ${MSVC}")
message(STATUS "UNIX: ${UNIX}")

# Set compiler-specific flags for Debug and Release builds using list variables
if(MSVC)
    # MSVC flags
    set(debug_flags_list "/Od" "/Zi")
    set(release_flags_list "/O2" "/DNDEBUG")
    set(linker_release_flags_list "/LTCG")
else()
    # For GCC/Clang, use -O0 and -g
    set(debug_flags_list "-Og" "-g")
    set(release_flags_list "-O3" "-ffast-math" "-DNDEBUG" "-march=native")
endif()

# Set CUDA flags if CUDA is enabled (as lists)
if(CMAKE_CUDA_COMPILER)
    if(MSVC)
        set(cuda_debug_flags_list "-G" "-g")
        set(cuda_release_flags_list "-O3" "--use_fast_math" "-DNDEBUG")
    else()
        set(cuda_debug_flags_list "-O0" "-G" "-g")
        set(cuda_release_flags_list "-O3" "--use_fast_math" "-DNDEBUG")
    endif()
endif()

# Define the executable target
add_executable(${PROJECT_NAME} ${SOURCES})

# Use Precompiled Header (PCH) from src/common.hpp
target_precompile_headers(${PROJECT_NAME} PRIVATE src/common.hpp)

# Apply compile options based on build configuration using generator expressions
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:DEBUG>:${debug_flags_list}>
    $<$<CONFIG:RELEASE>:${release_flags_list}>
)
if(CMAKE_CUDA_COMPILER)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:DEBUG>:${cuda_debug_flags_list}>
        $<$<CONFIG:RELEASE>:${cuda_release_flags_list}>
    )
endif()

# Link LibTorch libraries
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Append linker flags for MSVC Release builds
if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RELEASE>:${linker_release_flags_list}>)
endif()

# Conditionally link stdc++fs for UNIX systems with older GCC versions
if(UNIX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

# export compile flags to json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Include directories
include_directories(src)
if(CMAKE_CUDA_COMPILER)
    include_directories(libtorch_cuda/include)
elseif(UNIX)
    include_directories(libtorch_cpu_linux/include)
elseif(MSVC)
    include_directories(libtorch_cpu_win/include)
endif()

# Copy DLLs on Windows after build
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()
