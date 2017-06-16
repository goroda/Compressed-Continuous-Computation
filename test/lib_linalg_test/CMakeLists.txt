cmake_minimum_required (VERSION 2.6)


file(GLOB SRC_FILES *.c)
add_executable(linalg_test ${SRC_FILES})
target_link_libraries(linalg_test 
    -g 
    c3
    cutest 
    ${blas_lapack} 
    -lm)


