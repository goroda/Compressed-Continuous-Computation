cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)
if ( ${static_libs} )
    add_library(tensor ${SRC_FILES})
else ()
    add_library(tensor SHARED ${SRC_FILES})
    target_link_libraries(tensor -fPIC array linalg)
endif()

