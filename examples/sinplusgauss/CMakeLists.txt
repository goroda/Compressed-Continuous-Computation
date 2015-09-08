cmake_minimum_required (VERSION 2.6)


add_executable(sinplusgauss sinrank2.c)
target_link_libraries(sinplusgauss -g
    clinalg 
    funcs 
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



