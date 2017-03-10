cmake_minimum_required (VERSION 2.6)


add_executable(gauss gauss.c)
target_link_libraries(gauss -g
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


