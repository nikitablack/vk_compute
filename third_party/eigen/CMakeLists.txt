cmake_minimum_required(VERSION 3.25)

set(EIGEN_VERSION 3.4)
set(EIGEN_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/${EIGEN_VERSION}")

if (EXISTS "${EIGEN_SRC_DIR}/CMakeLists.txt")
    add_subdirectory("${EIGEN_SRC_DIR}")
else()
    message(STATUS "Downloading Eigen ${EIGEN_VERSION}")

    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://github.com/PX4/eigen.git
        GIT_TAG ${EIGEN_VERSION}
        SOURCE_DIR ${EIGEN_SRC_DIR}
    )

    FetchContent_MakeAvailable(eigen)
endif()
