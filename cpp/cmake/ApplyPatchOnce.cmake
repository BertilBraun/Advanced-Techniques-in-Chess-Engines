execute_process(
        COMMAND git apply --check "${PATCH}"
        WORKING_DIRECTORY "${REPOSITORY}"
        RESULT_VARIABLE applyCheckResult
        OUTPUT_QUIET
        ERROR_QUIET
)

if (applyCheckResult EQUAL 0)
    execute_process(
            COMMAND git apply --whitespace=nowarn "${PATCH}"
            WORKING_DIRECTORY "${REPOSITORY}"
            RESULT_VARIABLE applyResult
    )
    if (NOT applyResult EQUAL 0)
        message(FATAL_ERROR "Failed to apply ${PATCH}")
    endif ()
    return()
endif ()

execute_process(
        COMMAND git apply --reverse --check "${PATCH}"
        WORKING_DIRECTORY "${REPOSITORY}"
        RESULT_VARIABLE reverseCheckResult
        OUTPUT_QUIET
        ERROR_QUIET
)
if (NOT reverseCheckResult EQUAL 0)
    message(FATAL_ERROR "${PATCH} is neither applicable nor already applied")
endif ()
