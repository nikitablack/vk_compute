cmake_minimum_required(VERSION 3.25)

set(CATCH_VERSION v3.14.0)
set(CATCH_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/${CATCH_VERSION}")

if (EXISTS "${CATCH_SRC_DIR}/CMakeLists.txt")
    add_subdirectory("${CATCH_SRC_DIR}")
else()
    message(STATUS "Downloading Catch2 ${CATCH_VERSION}")

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG ${CATCH_VERSION}
        SOURCE_DIR ${CATCH_SRC_DIR}
    )

    FetchContent_MakeAvailable(Catch2)
endif()

list(APPEND CMAKE_MODULE_PATH ${CATCH_SRC_DIR}/extras)
include(CTest)
include(Catch)
