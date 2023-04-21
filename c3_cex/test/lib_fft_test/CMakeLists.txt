cmake_minimum_required (VERSION 3.20)


file(GLOB SRC_FILES fft_test.c main.c)

add_executable(fft_test ${SRC_FILES})
target_link_libraries(fft_test
    -g 
    c3
    cutest 
    ${blas_lapack} 
    -lm)

