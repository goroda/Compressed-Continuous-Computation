cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES 
  #cross2d.c
  diffusion.c
  dmrgprod.c
  dmrg.c
  indmanage.c
  ft.c
  qmarray.c
  quasimatrix.c)

if ( ${static_libs} )
    add_library(clinalg ${SRC_FILES})
else ()
    add_library(clinalg SHARED ${SRC_FILES})
    target_link_libraries(clinalg 
      -fPIC 
      array 
      funcs 
      linalg 
      ${blas_lapack})
endif() 

