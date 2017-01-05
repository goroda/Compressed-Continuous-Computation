cmake_minimum_required (VERSION 2.6)


add_executable(alsregress alsregress.c)
target_link_libraries(alsregress -g
    clinalg 
    funcs 
    opt
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



