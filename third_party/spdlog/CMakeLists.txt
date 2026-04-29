cmake_minimum_required(VERSION 3.25)

set(SPDLOG_VERSION v1.17.0)
set(SPDLOG_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/${SPDLOG_VERSION}")

if (EXISTS "${SPDLOG_SRC_DIR}/CMakeLists.txt")
    add_subdirectory("${SPDLOG_SRC_DIR}")
else()
    message(STATUS "Downloading spdlog ${SPDLOG_VERSION}")

    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG ${SPDLOG_VERSION}
        SOURCE_DIR ${SPDLOG_SRC_DIR}
    )

    FetchContent_MakeAvailable(spdlog)
endif()
