cmake_minimum_required (VERSION 2.6)


add_executable(elliptic1d elliptic.c)
target_link_libraries(elliptic1d 
  -g
  approx_interface
    prob
    clinalg 
    funcs
    opt 
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



