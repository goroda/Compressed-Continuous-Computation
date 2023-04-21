cmake_minimum_required (VERSION 3.20)


file(GLOB SRC_FILES *.c)
add_executable(probability_test ${SRC_FILES})
target_link_libraries(probability_test 
  -g
  c3
  cutest 
  ${blas_lapack} 
  -lm)


