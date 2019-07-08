cmake_minimum_required (VERSION 2.6)


file(GLOB SRC_FILES *.c)
add_executable(clinalg_test2 ${SRC_FILES})
target_link_libraries(clinalg_test2
    clinalg 
    funcs 
    quadrature 
    linalg 
    array 
    stringmanip
    cutest 
    ${blas_lapack} 
    -lm)


