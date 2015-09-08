cmake_minimum_required (VERSION 2.6)


file(GLOB SRC_FILES *.c)
add_executable(funcs_test ${SRC_FILES})
target_link_libraries(funcs_test 
    -g 
    funcs 
    quadrature 
    array 
    stringmanip
    cutest 
    ${blas_lapack} 
    -lm)


