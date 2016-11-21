cmake_minimum_required (VERSION 2.6)


file(GLOB SRC_FILES other_test.c piecewise_test.c linelm_test.c poly_test.c testfunctions.c main.c)

add_executable(funcs_test ${SRC_FILES})
target_link_libraries(funcs_test 
    -g 
    funcs
    opt 
    quadrature 
    array 
    stringmanip
    cutest 
    ${blas_lapack} 
    -lm)


