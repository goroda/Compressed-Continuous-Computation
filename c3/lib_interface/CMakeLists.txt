cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES 
  approximate.c
)

if ( ${static_libs} )
    add_library(approx_interface ${SRC_FILES})
else ()
    add_library(approx_interface SHARED ${SRC_FILES})
    target_link_libraries(approx_interface
      -fPIC 
      array 
      funcs 
      linalg
      clinalg
      suplearn
      ${blas_lapack})
endif() 
