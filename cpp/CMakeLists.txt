cmake_minimum_required(VERSION 3.23.3 FATAL_ERROR)
project(AlphaZeroSelfPlay LANGUAGES CXX)

# Enable position-independent code for all targets
# TODO? set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    # set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cuda/share/cmake/Torch")
elseif(UNIX)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux")
    # set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux/share/cmake/Torch")
elseif(MSVC)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_win")
    # set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_win/share/cmake/Torch")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}") #  -fPIC -D_GLIBCXX_USE_CXX11_ABI=1
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CUDA_FLAGS}")

# --- Define additional LibTorch libraries explicitly ---
if(UNIX)
    # Make sure to point to libtorch.so. Adjust the path if needed.
    # set(ADDITIONAL_TORCH_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux/lib/libtorch.so;${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux/lib/libc10.so;${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux/lib/libtorch_python.so;${CMAKE_CURRENT_SOURCE_DIR}/libtorch_cpu_linux/lib/libtorch_global_deps.so")
endif()


# --- Add tensorboard_logger via FetchContent ---
# find_package(Protobuf REQUIRED)
include(FetchContent)
# FetchContent_Declare(
#     tensorboard_logger
#     GIT_REPOSITORY https://github.com/RustingSword/tensorboard_logger.git
#     GIT_TAG master
# )
# FetchContent_MakeAvailable(tensorboard_logger)

# --- Add fmtlib via FetchContent ---
# FetchContent_Declare(
#     fmt
#     GIT_REPOSITORY https://github.com/fmtlib/fmt.git
#     GIT_TAG 8.1.1  # Or another version tag
# )
# FetchContent_MakeAvailable(fmt)

# --- FetchContent for pybind11 ---
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6  # Use the desired version
)
FetchContent_MakeAvailable(pybind11)


# --- Include directories ---
include_directories(src)
# if(CMAKE_CUDA_COMPILER)
#     include_directories(libtorch_cuda/include)
# elseif(UNIX)
#    include_directories(libtorch_cpu_linux/include)
# elseif(MSVC)
#    include_directories(libtorch_cpu_win/include)
# endif()

# --- Collect sources for the Python module ---
# Gather all .cpp files under src/
file(GLOB_RECURSE MODULE_SOURCES "src/*.cpp")

# Precompiled headers for src/common.hpp
#TODO! set(PCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/common.hpp")

# --- Build the Python extension module ---
pybind11_add_module(AlphaZeroCpp ${MODULE_SOURCES})

# TODO: # Build src/main.cpp into an executable
# TODO: # add all .cpp files in src/ to the executable
# TODO: file(GLOB_RECURSE EXECUTABLE_SOURCES "src/*.cpp")
# TODO: # remove src/binding.cpp from the list of sources
# TODO: list(REMOVE_ITEM EXECUTABLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/binding.cpp")
# TODO: add_executable(AlphaZeroSelfPlay ${EXECUTABLE_SOURCES})

# --- Set compile options (optional) ---
#TODO! if(MSVC)
#TODO!     set(debug_flags_list "/Od" "/Zi")
#TODO!     set(release_flags_list "/O2" "/DNDEBUG")
#TODO! else()
#TODO!     set(debug_flags_list "-Og" "-g")
#TODO!     set(release_flags_list "-O3" "-ffast-math" "-DNDEBUG" "-march=native")
#TODO! endif()
#TODO! 
#TODO! target_compile_options(AlphaZeroCpp PRIVATE
#TODO!     $<$<CONFIG:DEBUG>:${debug_flags_list}>
#TODO!     $<$<CONFIG:RELEASE>:${release_flags_list}>
#TODO! )
#TODO! if(CMAKE_CUDA_COMPILER)
#TODO!     if(MSVC)
#TODO!         set(cuda_debug_flags_list "-G" "-g")
#TODO!         set(cuda_release_flags_list "-O3" "--use_fast_math" "-DNDEBUG")
#TODO!     else()
#TODO!         set(cuda_debug_flags_list "-O0" "-G" "-g")
#TODO!         set(cuda_release_flags_list "-O3" "--use_fast_math" "-DNDEBUG")
#TODO!     endif()
#TODO!     target_compile_options(AlphaZeroCpp PRIVATE
#TODO!         $<$<CONFIG:DEBUG>:${cuda_debug_flags_list}>
#TODO!         $<$<CONFIG:RELEASE>:${cuda_release_flags_list}>
#TODO!     )
#TODO! endif()


# --- Link libraries ---
target_link_libraries(AlphaZeroCpp PRIVATE "${TORCH_LIBRARIES}") #  tensorboard_logger fmt::fmt TODO?  ${ADDITIONAL_TORCH_LIBS}

# if(MSVC)
#     target_link_options(AlphaZeroCpp PRIVATE $<$<CONFIG:RELEASE>:"/LTCG">)
# endif()

# --- Embed RPATH so that LibTorch libraries can be found at runtime ---
# if(UNIX)
#     # Assume that you copy the LibTorch shared libraries (like libc10.so, libtorch.so) to the same directory
#     # as your Python module. Get that directory's absolute path.
#     get_filename_component(PY_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../cpp_py" ABSOLUTE)
#     message(STATUS "Python module directory (for RPATH): ${PY_MODULE_DIR}")
# 
#     # Set both BUILD_RPATH and INSTALL_RPATH to that directory.
#     set_target_properties(AlphaZeroCpp PROPERTIES BUILD_RPATH "${PY_MODULE_DIR}")
#     set_target_properties(AlphaZeroCpp PROPERTIES INSTALL_RPATH "${PY_MODULE_DIR}")
# endif()

set_property(TARGET AlphaZeroCpp PROPERTY CXX_STANDARD 20)


set_target_properties(AlphaZeroCpp PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    INSTALL_RPATH "$ORIGIN")

# TODO: target_compile_options(AlphaZeroSelfPlay PRIVATE
# TODO:     $<$<CONFIG:DEBUG>:${debug_flags_list}>
# TODO:     $<$<CONFIG:RELEASE>:${release_flags_list}>
# TODO: )
# TODO: if(CMAKE_CUDA_COMPILER)
# TODO:     if(MSVC)
# TODO:         set(cuda_debug_flags_list "-G" "-g")
# TODO:         set(cuda_release_flags_list "-O3" "--use_fast_math" "-DNDEBUG")
# TODO:     else()
# TODO:         set(cuda_debug_flags_list "-O0" "-G" "-g")
# TODO:         set(cuda_release_flags_list "-O3" "--use_fast_math" "-DNDEBUG")
# TODO:     endif()
# TODO:     target_compile_options(AlphaZeroSelfPlay PRIVATE
# TODO:         $<$<CONFIG:DEBUG>:${cuda_debug_flags_list}>
# TODO:         $<$<CONFIG:RELEASE>:${cuda_release_flags_list}>
# TODO:     )
# TODO: endif()
# TODO: 

# TODO: target_link_libraries(AlphaZeroSelfPlay PRIVATE "${TORCH_LIBRARIES}" ${ADDITIONAL_TORCH_LIBS} c10 tensorboard_logger fmt::fmt)
# TODO: 

# TODO: if(MSVC)
# TODO:     target_link_options(AlphaZeroSelfPlay PRIVATE $<$<CONFIG:RELEASE>:"/LTCG">)
# TODO: endif()
# TODO: if(UNIX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
# TODO:     target_link_libraries(AlphaZeroSelfPlay stdc++fs)
# TODO: endif()


# --- Export compile commands ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Post-build: Copy the generated shared library to "../../cpp_py/" ---
add_custom_command(TARGET AlphaZeroCpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/../../cpp_py/"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:AlphaZeroCpp> "${CMAKE_CURRENT_BINARY_DIR}/../../cpp_py/"
    COMMENT "Copying AlphaZeroCpp module to ../../cpp_py/"
)
