cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES
    constelm.c
    polynomials.c
    piecewisepoly.c
    hpoly.c
    fourier.c
    linelm.c
    kernels.c
    fapprox.c
    funcs_mixed.c
    funcs_not_standard.c
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
  if (MAKE_PYTHON_WRAPPERS)
      include_directories(${PYTHON_INCLUDE_DIRS})
      add_library(c3 SHARED ${SRC_FILES})
      target_link_libraries(c3
        -fPIC
        opt
        quadrature 
        array
        stringmanip
        ${python_lib}
        -lm
        )
  else()
    target_link_libraries(funcs 
        -fPIC
        opt
        quadrature 
        array
        stringmanip
        -lm
        )
endif() 

