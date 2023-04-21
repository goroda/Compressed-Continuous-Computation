cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)

if ( ${static_libs} )
    add_library(opt ${SRC_FILES})
else ()
    add_library(opt SHARED ${SRC_FILES})
    target_link_libraries(opt -fPIC  array linalg ${blas_lapack})
endif() 

