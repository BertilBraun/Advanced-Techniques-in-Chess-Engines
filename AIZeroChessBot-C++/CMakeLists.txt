cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(AIZeroChessBot)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)

cmake_policy(SET CMP0054 NEW)

# LibTorch configuration
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Automatically include all source files from the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Optional: Specify include directories if you have header files in different locations
include_directories(src)
include_directories(libtorch/include)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET AIZeroChessBot
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:AIZeroChessBot>)
endif (MSVC)