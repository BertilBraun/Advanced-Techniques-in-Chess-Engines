cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(AIZeroChessBot)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

cmake_policy(SET CMP0054 NEW)

# LibTorch configuration
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Automatically include all source files from the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp")
# file(GLOB_RECURSE SOURCES "src/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Optional: Specify include directories if you have header files in different locations
include_directories(src)
