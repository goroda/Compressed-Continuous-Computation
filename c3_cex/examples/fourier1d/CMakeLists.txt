cmake_minimum_required (VERSION 2.6)


add_executable(fourier1d fourier1d.c)
target_link_libraries(fourier1d
    -g 
    c3
    ${blas_lapack} 
    -lm)

