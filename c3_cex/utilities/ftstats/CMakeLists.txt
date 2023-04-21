cmake_minimum_required (VERSION 2.6)

add_executable(ftstats ftstats.c)
target_link_libraries(ftstats
    -g
    c3
    ${blas_lapack}
    -lm)

  # install(TARGETS ftstats EXPORT ftstats DESTINATION "${bin_dest}")
install(TARGETS ftstats DESTINATION bin)
