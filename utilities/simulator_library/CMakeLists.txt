cmake_minimum_required (VERSION 2.6)

add_executable(simlib_util simulator.c functions.c)
target_link_libraries(simlib_util
    -g
    c3
    ${blas_lapack}
    -lm)

  # install(TARGETS simlib_util EXPORT simlib_util DESTINATION "${bin_dest}")
install(TARGETS simlib_util DESTINATION bin)




