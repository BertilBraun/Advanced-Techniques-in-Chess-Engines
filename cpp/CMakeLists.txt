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

add_library(
        az_go
        src/games/go/GoEncoding.cpp
        src/games/go/GoState.cpp
        src/games/go/GoSymmetry.cpp
)
target_include_directories(az_go PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_precompile_headers(az_go PRIVATE src/common.hpp)
az_enable_warnings(az_go)

if (ENABLE_NATIVE_ARCHITECTURE AND NOT MSVC)
    target_compile_options(az_go PRIVATE -march=native)
endif ()

if (BUILD_TESTING)
    add_executable(GoStateTest test/TestGoState.cpp)
    target_link_libraries(GoStateTest PRIVATE az_go)
    target_precompile_headers(GoStateTest PRIVATE src/common.hpp)
    az_enable_warnings(GoStateTest)
    add_test(NAME GoStateTest COMMAND GoStateTest)

    add_executable(FixedPuctTest test/TestFixedPuct.cpp)
    target_link_libraries(FixedPuctTest PRIVATE az_go)
    target_precompile_headers(FixedPuctTest PRIVATE src/common.hpp)
    az_enable_warnings(FixedPuctTest)
    add_test(NAME FixedPuctTest COMMAND FixedPuctTest)
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
    target_link_libraries(az_go_native PRIVATE az_go)
    target_precompile_headers(az_go_native PRIVATE src/common.hpp)
    az_enable_warnings(az_go_native)
endif ()
