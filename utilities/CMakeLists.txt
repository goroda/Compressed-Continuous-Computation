cmake_minimum_required (VERSION 2.6)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(simulator_library)
add_subdirectory(gen_random_samples)
add_subdirectory(ftstats)
add_subdirectory(regress)
add_subdirectory(precompute_recursion)
