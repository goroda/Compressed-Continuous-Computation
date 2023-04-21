cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)
if (${static_libs} )
    add_library(stringmanip ${SRC_FILES})
else ()
    add_library(stringmanip SHARED ${SRC_FILES})
endif()

