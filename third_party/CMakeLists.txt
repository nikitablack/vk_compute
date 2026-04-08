option(BUILD_TESTS "Whether or not to build the tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(catch SYSTEM)
endif()

add_subdirectory(cmakerc SYSTEM)
add_subdirectory(fmt SYSTEM)
add_subdirectory(vma SYSTEM)
