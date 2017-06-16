cmake_minimum_required (VERSION 2.6)

add_executable(array_test lib_array_test.c)
TARGET_LINK_LIBRARIES(array_test 
    -g 
    c3
    cutest 
    -lm)

