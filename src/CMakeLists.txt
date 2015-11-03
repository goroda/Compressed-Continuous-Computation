cmake_minimum_required (VERSION 2.6)

add_subdirectory(lib_stringmanip)
add_subdirectory(lib_array)
add_subdirectory(lib_linalg)
add_subdirectory(lib_clinalg)
add_subdirectory(lib_quadrature)
add_subdirectory(lib_funcs)
add_subdirectory(lib_twodfuncs)
add_subdirectory(lib_tensor)
add_subdirectory(lib_tensdecomp)
add_subdirectory(lib_optimization)
add_subdirectory(lib_probability)


#create a libray containing everything!
file(GLOB SRC_FILES */*.c)
if (${static_libs} )
    add_library(c3 ${SRC_FILES})
else ()
    add_library(c3 SHARED ${SRC_FILES})
    target_link_libraries(c3
        ${blas_lapack}
        -lm
        )
        
endif()

