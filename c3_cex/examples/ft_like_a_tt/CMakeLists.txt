cmake_minimum_required (VERSION 2.6)

add_executable(ft_like_a_tt ft_like_a_tt.c)
target_link_libraries(ft_like_a_tt -g
    c3
    ${blas_lapack}
    -lm)

