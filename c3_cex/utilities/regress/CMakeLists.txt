cmake_minimum_required (VERSION 2.6)


add_executable(ftregress regress.c)
target_link_libraries(ftregress -g
  c3
  ${blas_lapack}
  -lm)



# install(TARGETS ftregress EXPORT ftregress DESTINATION "${bin_dest}")
install(TARGETS ftregress DESTINATION bin)
