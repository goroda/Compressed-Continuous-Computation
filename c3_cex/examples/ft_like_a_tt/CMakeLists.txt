cmake_minimum_required (VERSION 3.20)

add_executable(ft_like_a_tt ft_like_a_tt.c)
target_link_libraries(ft_like_a_tt -g
    c3
    ${blas_lapack}
    -lm)

