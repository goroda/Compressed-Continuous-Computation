cmake_minimum_required (VERSION 2.6)


add_executable(genregress genregress.c)
target_link_libraries(genregress -g
    clinalg 
    funcs 
    opt
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



