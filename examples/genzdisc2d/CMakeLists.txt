cmake_minimum_required (VERSION 2.6)

#SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fsanitize=address" )
add_executable(genz2d genz2d.c)
target_link_libraries(genz2d -g
    clinalg 
    funcs 
    opt
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



