cmake_minimum_required (VERSION 3.20)


add_executable(fourier1d fourier1d.c)
target_link_libraries(fourier1d
    -g 
    c3
    ${blas_lapack} 
    -lm)

