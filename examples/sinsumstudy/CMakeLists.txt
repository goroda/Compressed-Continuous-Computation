cmake_minimum_required (VERSION 2.6)


add_executable(sinsumstudy sinsum.c)
target_link_libraries(sinsumstudy -g 
    clinalg 
    funcs 
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)


