cmake_minimum_required (VERSION 2.6)

add_executable(random_sample random.c)
target_link_libraries(random_sample
    -g
    c3
    ${blas_lapack}
    -lm)

# install(TARGETS random_sample EXPORT random_sample DESTINATION "${bin_dest}")
install(TARGETS random_sample DESTINATION bin)
