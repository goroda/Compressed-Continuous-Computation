cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)
add_library(cutest ${SRC_FILES})

