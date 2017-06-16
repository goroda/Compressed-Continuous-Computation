cmake_minimum_required (VERSION 2.6)


add_executable(aioregress aioregress.c)
target_link_libraries(aioregress -g
    c3
    ${blas_lapack}
    -lm)



