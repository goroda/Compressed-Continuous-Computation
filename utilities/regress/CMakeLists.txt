cmake_minimum_required (VERSION 2.6)


add_executable(ftregress regress.c)
target_link_libraries(ftregress -g
    clinalg 
    funcs 
    opt
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



