cmake_minimum_required (VERSION 2.6)

add_executable(precompute_recursion_coeff recur.c)
target_link_libraries(precompute_recursion_coeff
    -lm)

# install(TARGETS random_sample EXPORT random_sample DESTINATION "${bin_dest}")
install(TARGETS precompute_recursion_coeff DESTINATION bin)
