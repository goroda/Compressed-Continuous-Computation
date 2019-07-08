cmake_minimum_required (VERSION 2.6)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/profiling/regress/bin)
add_subdirectory(all_in_one)
add_subdirectory(als)
