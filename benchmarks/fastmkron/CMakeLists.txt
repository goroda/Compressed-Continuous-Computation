cmake_minimum_required (VERSION 2.6)


add_executable(fmkronbench fmkron.c)
target_link_libraries(fmkronbench -g
    clinalg 
    funcs
    opt 
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



