cmake_minimum_required (VERSION 2.6)


file(GLOB SRC_FILES *.c)
add_executable(probability_test ${SRC_FILES})
target_link_libraries(probability_test 
    -g 
    prob
    #opt
    clinalg
    funcs
    quadrature
    linalg 
    array 
    stringmanip
    cutest 
    ${blas_lapack} 
    -lm)


