cmake_minimum_required (VERSION 2.6)


add_executable(dmrgprodbench dmrgprod.c)
target_link_libraries(dmrgprodbench -g
    clinalg 
    funcs 
    opt
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



