cmake_minimum_required (VERSION 2.6)


file(GLOB SRC_FILES *.c)
add_executable(clinalg_test ${SRC_FILES})
target_link_libraries(clinalg_test 
    clinalg 
    funcs 
    quadrature 
    linalg 
    array 
    stringmanip
    cutest 
    ${blas_lapack} 
    -lm)


