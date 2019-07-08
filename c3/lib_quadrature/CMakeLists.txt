cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)

if ( ${static_libs} )
    add_library(quadrature ${SRC_FILES})
    target_link_libraries(quadrature -fPIC linalg)
else ()
    add_library(quadrature SHARED ${SRC_FILES} )
    target_link_libraries(quadrature -fPIC linalg array)
endif() 

