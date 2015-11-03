cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES 
    twodfunctions.c
    )

if ( ${static_libs} )
    add_library(funcs2 ${SRC_FILES})
    target_link_libraries(funcs2 )
else ()
    add_library(funcs2 SHARED ${SRC_FILES} )
    target_link_libraries(funcs2
        funcs
        clinalg
        -fPIC 
        quadrature 
        array
        stringmanip
        -lm
        )
endif() 

