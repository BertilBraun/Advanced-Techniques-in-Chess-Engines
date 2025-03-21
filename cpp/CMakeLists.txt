cmake_minimum_required(VERSION 3.23.3 FATAL_ERROR)
project(AIZeroChessBot LANGUAGES CXX)

# Determine if CUDA is available and set languages accordingly
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif()


# LibTorch configuration
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cuda")
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cuda/share/cmake/Torch")
elseif(UNIX)
    # if on linux
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux")
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux/share/cmake/Torch")
elseif(MSVC)
    # if on windows
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_win")
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_win/share/cmake/Torch")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()
find_package(Torch REQUIRED)

# Automatically include all source files from the src directory
file(GLOB_RECURSE SOURCES "src/main.cpp")

# General flags for both debug and release
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Release-specific flags
if(NOT MSVC)
    # GCC, Clang, and other UNIX-like compilers
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    # desirable: set(CMAKE_CXX_FLAGS_RELEASE "-O3 --fast-math -DNDEBUG -march=native")
    if(CMAKE_CUDA_COMPILER)
        # Setting CUDA flags using string APPEND to avoid overriding existing flags
        # desirable: string(APPEND CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
    endif()
else()
    # MSVC specific flags
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
endif()


# Ensure we're setting the correct flags for the target rather than globally
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RELEASE>:${CMAKE_CXX_FLAGS_RELEASE}>)
if(CMAKE_CUDA_COMPILER)
    # desirable: target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RELEASE>:${CMAKE_CUDA_FLAGS_RELEASE}>)
endif()

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Conditional linking against stdc++fs if not on MSVC and less than GCC 9
if(NOT MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

# Optional: Specify include directories if you have header files in different locations
include_directories(src)
if (CMAKE_CUDA_COMPILER)
    include_directories(libtorch_cuda/include)
elseif(UNIX)
    include_directories(libtorch_cpu_linux/include)
elseif(MSVC)
    include_directories(libtorch_cpu_win/include)
endif()

# Copy DLLs on Windows
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif(MSVC)
