cmake_minimum_required (VERSION 2.6)


add_executable(alsregress alsregress.c)
target_link_libraries(alsregress -g
    c3
    ${blas_lapack}
    -lm)



