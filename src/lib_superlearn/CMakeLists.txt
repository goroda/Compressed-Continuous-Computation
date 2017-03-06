cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES 
  regress.c
  )

if ( ${static_libs} )
    add_library(suplearn ${SRC_FILES})
else ()
    add_library(suplearn SHARED ${SRC_FILES})
    target_link_libraries(suplearn 
      -fPIC 
      array 
      funcs 
      linalg
      clinalg
      ${blas_lapack})
endif() 

