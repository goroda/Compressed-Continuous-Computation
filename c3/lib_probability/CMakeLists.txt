cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)

if ( ${static_libs} )
    add_library(prob ${SRC_FILES})
else ()
    add_library(prob SHARED ${SRC_FILES})
    target_link_libraries(prob 
        -fPIC 
        array 
        linalg 
        funcs 
        clinalg 
        opt 
        ${blas_lapack})
endif() 

