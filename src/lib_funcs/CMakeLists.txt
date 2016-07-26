cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES 
    polynomials.c
    piecewisepoly.c
    hpoly.c
    linelm.c
    fapprox.c
    functions.c
    monitoring.c
    space.c
    fwrap.c
    pivoting.c)

if ( ${static_libs} )
    add_library(funcs ${SRC_FILES})
    target_link_libraries(funcs )
else ()
    add_library(funcs SHARED ${SRC_FILES} )
    target_link_libraries(funcs 
        -fPIC 
        quadrature 
        array
        stringmanip
        -lm
        )
endif() 

