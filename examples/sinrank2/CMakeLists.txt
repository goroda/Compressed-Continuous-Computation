cmake_minimum_required (VERSION 2.6)


add_executable(sinrank2 sinrank2.c)
target_link_libraries(sinrank2 -g
    clinalg 
    funcs 
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



