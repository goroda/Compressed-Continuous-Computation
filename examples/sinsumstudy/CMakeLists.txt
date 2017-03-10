cmake_minimum_required (VERSION 2.6)


add_executable(sinsumstudy sinsum.c)
target_link_libraries(sinsumstudy -g
  approx_interface
    clinalg 
    funcs 
    opt
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)


