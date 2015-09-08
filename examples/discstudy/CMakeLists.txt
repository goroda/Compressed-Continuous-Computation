cmake_minimum_required (VERSION 2.6)

#SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fsanitize=address" )
add_executable(genznd genznd.c)
target_link_libraries(genznd -g
    clinalg 
    funcs 
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)



