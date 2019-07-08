cmake_minimum_required (VERSION 2.6)

add_executable(string_test lib_string_test.c)
TARGET_LINK_LIBRARIES(string_test -g c3 cutest -lm)

