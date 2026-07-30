add_library(
        az_v2_go
        games/go/GoEncoding.cpp
        games/go/GoState.cpp
        games/go/GoSymmetry.cpp
)
target_include_directories(az_v2_go PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(az_v2_go PUBLIC cxx_std_20)
set_target_properties(az_v2_go PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    target_compile_options(az_v2_go PRIVATE /W4 /WX)
else ()
    target_compile_options(az_v2_go PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif ()

if (BUILD_TESTING)
    add_executable(GoStateV2Test ../../test/v2/TestGoState.cpp)
    target_link_libraries(GoStateV2Test PRIVATE az_v2_go)
    if (MSVC)
        target_compile_options(GoStateV2Test PRIVATE /W4 /WX)
    else ()
        target_compile_options(GoStateV2Test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif ()
    add_test(NAME GoStateV2Test COMMAND GoStateV2Test)

    add_executable(FixedPuctV2Test ../../test/v2/TestFixedPuct.cpp)
    target_link_libraries(FixedPuctV2Test PRIVATE az_v2_go)
    if (MSVC)
        target_compile_options(FixedPuctV2Test PRIVATE /W4 /WX)
    else ()
        target_compile_options(FixedPuctV2Test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif ()
    add_test(NAME FixedPuctV2Test COMMAND FixedPuctV2Test)
endif ()

option(AZ_BUILD_V2_PYTHON "Build the focused v2 Go Python binding" OFF)
if (AZ_BUILD_V2_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    include(FetchContent)
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4
    )
    FetchContent_MakeAvailable(pybind11)
    pybind11_add_module(
            az_go_native
            bindings/module.cpp
            bindings/go_bindings.cpp
            bindings/search_bindings.cpp
    )
    target_link_libraries(az_go_native PRIVATE az_v2_go)
    if (MSVC)
        target_compile_options(az_go_native PRIVATE /W4 /WX)
    else ()
        target_compile_options(az_go_native PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif ()
endif ()
