cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)

if ( ${static_libs} )
    add_library(tensdecomp ${SRC_FILES})
    target_link_libraries(tensdecomp tensor)
else ()
    add_library(tensdecomp SHARED ${SRC_FILES})
    target_link_libraries(tensdecomp -fPIC tensor)
endif()

