cmake_minimum_required (VERSION 2.6)


add_executable(simlib simlib.c)
target_link_libraries(simlib -g
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



