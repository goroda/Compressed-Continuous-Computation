cmake_minimum_required (VERSION 2.6)

add_executable(random_sample random.c)
target_link_libraries(random_sample
    -g
    array
    stringmanip
    ${blas_lapack}
    -lm)



