cmake_minimum_required (VERSION 3.20)

file(GLOB SRC_FILES *.c)
add_library(cutest ${SRC_FILES})

