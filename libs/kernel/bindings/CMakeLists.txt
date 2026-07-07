cmake_minimum_required(VERSION 3.25)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(python_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)

install(TARGETS
            python_kernel
        LIBRARY DESTINATION
            .
)

target_link_libraries(python_kernel
    PRIVATE
        libs::kernel
        libs::kernel::minmax
        libs::utils
)

set_target_properties(python_kernel
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
