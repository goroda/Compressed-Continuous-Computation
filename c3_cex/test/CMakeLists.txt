cmake_minimum_required (VERSION 2.6)

#include_directories(/opt/local/include)

#add_executable(allTests CuTest.c vec_test.c mat_test.c 
#                        linalg_test.c tensor_test.c tt_test.c cross_test.c AllMyTests.c)
#TARGET_LINK_LIBRARIES(allTests -g array -lm)


include_directories(CuTest)
add_subdirectory(CuTest)

add_subdirectory(lib_stringmanip_test)
add_subdirectory(lib_array_test)

add_subdirectory(lib_linalg_test)
add_subdirectory(lib_fft_test)

add_subdirectory(lib_quadrature_test)
add_subdirectory(lib_funcs_test)

add_subdirectory(lib_clinalg_test)
add_subdirectory(lib_superlearn_test)
add_subdirectory(lib_optimization_test)

# add_subdirectory(lib_clinalg_test2)

# add_subdirectory(lib_tensor_test)
# add_subdirectory(lib_tensdecomp_test)

add_subdirectory(lib_probability_test)

