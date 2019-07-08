cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)

if ( ${static_libs} )
    add_library(linalg ${SRC_FILES})
else ()
    add_library(linalg SHARED ${SRC_FILES})
    target_link_libraries(linalg -fPIC array ${blas_lapack})
endif() 

