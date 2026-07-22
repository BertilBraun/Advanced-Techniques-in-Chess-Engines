execute_process(
        COMMAND git apply --check "${PATCH}"
        WORKING_DIRECTORY "${REPOSITORY}"
        RESULT_VARIABLE apply_check_result
        OUTPUT_QUIET
        ERROR_QUIET
)

if (apply_check_result EQUAL 0)
    execute_process(
            COMMAND git apply --whitespace=nowarn "${PATCH}"
            WORKING_DIRECTORY "${REPOSITORY}"
            RESULT_VARIABLE apply_result
    )
    if (NOT apply_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply ${PATCH}")
    endif ()
    return()
endif ()

execute_process(
        COMMAND git apply --reverse --check "${PATCH}"
        WORKING_DIRECTORY "${REPOSITORY}"
        RESULT_VARIABLE reverse_check_result
        OUTPUT_QUIET
        ERROR_QUIET
)
if (NOT reverse_check_result EQUAL 0)
    message(FATAL_ERROR "${PATCH} is neither applicable nor already applied")
endif ()
