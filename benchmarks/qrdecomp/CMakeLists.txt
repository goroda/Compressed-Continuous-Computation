cmake_minimum_required (VERSION 2.6)


add_executable(qrbench qr.c)
target_link_libraries(qrbench -g
    clinalg 
    funcs 
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



