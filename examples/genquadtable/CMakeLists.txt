cmake_minimum_required (VERSION 2.6)


add_executable(qtable qtable.c)
target_link_libraries(qtable -g 
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)


