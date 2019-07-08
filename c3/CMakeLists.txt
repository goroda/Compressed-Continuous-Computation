cmake_minimum_required (VERSION 3.10)

if (MAKE_PYTHON_WRAPPERS)
  add_definitions(-DCOMPILE_WITH_PYTHON)
endif()

if (BUILD_SUB_LIBS)
  add_subdirectory(lib_stringmanip)
  add_subdirectory(lib_array)
  add_subdirectory(lib_linalg)
  add_subdirectory(lib_optimization)
  add_subdirectory(lib_fft)
  add_subdirectory(lib_quadrature)
  add_subdirectory(lib_funcs)

  add_subdirectory(lib_clinalg)
  add_subdirectory(lib_superlearn)
  add_subdirectory(lib_interface)

  #add_subdirectory(lib_twodfuncs)
  add_subdirectory(lib_tensor)
  add_subdirectory(lib_tensdecomp)

  add_subdirectory(lib_probability)

endif()

#########################################################

#set headers
set(header lib_stringmanip/stringmanip.h
           lib_array/array.h
           lib_superlearn/learning_options.h
           lib_superlearn/objective_functions.h
           lib_superlearn/superlearn_util.h
           lib_superlearn/superlearn.h           
           lib_superlearn/parameterization.h
           lib_superlearn/regress.h
           lib_interface/c3_interface.h
           lib_interface/approximate.h
           lib_clinalg/diffusion.h
           lib_clinalg/dmrg.h
           lib_clinalg/dmrgprod.h
           lib_clinalg/ft.h
           lib_clinalg/indmanage.h
           lib_clinalg/qmarray.h
           lib_clinalg/quasimatrix.h
           lib_clinalg/lib_clinalg.h
           lib_fft/fft.h
           lib_funcs/fapprox.h
           lib_funcs/functions.h
           lib_funcs/fwrap.h
           lib_funcs/hpoly.h
           lib_funcs/fourier.h
           lib_funcs/legtens.h
           lib_funcs/lib_funcs.h
           lib_funcs/linelm.h
           lib_funcs/constelm.h
           lib_funcs/kernels.h
           lib_funcs/monitoring.h
           lib_funcs/piecewisepoly.h
           lib_funcs/pivoting.h
           lib_funcs/polynomials.h
           lib_funcs/space.h
           lib_linalg/lib_linalg.h
           lib_linalg/linalg.h
           lib_linalg/matrix_util.h
           lib_optimization/lib_optimization.h
           lib_optimization/optimization.h
           lib_quadrature/legquadrules.h
           lib_quadrature/lib_quadrature.h
           lib_quadrature/quadrature.h
           lib_tensdecomp/candecomp.h
           lib_tensdecomp/cross.h
           lib_tensdecomp/tensortrain.h
           lib_tensdecomp/tt_integrate.h
           lib_tensdecomp/tt_multilinalg.h
           lib_tensor/tensor.h
           lib_probability/lib_probability.h
           lib_probability/probability.h
           lib_probability/probability_utils.h
           # ../include/c3.h
           )


set(src lib_array/array.c
        lib_superlearn/learning_options.c
        lib_superlearn/objective_functions.c
        lib_superlearn/parameterization.c
        lib_superlearn/regress.c
        lib_superlearn/superlearn.c
        lib_superlearn/superlearn_util.c
        lib_interface/approximate.c
        lib_clinalg/diffusion.c
        lib_clinalg/dmrg.c
        lib_clinalg/dmrgprod.c
        lib_clinalg/ft.c
        lib_clinalg/indmanage.c
        lib_clinalg/iterative.c
        lib_clinalg/qmarray.c
        lib_clinalg/quasimatrix.c
        lib_fft/fft.c
        lib_funcs/fapprox.c
        lib_funcs/functions.c
        lib_funcs/fwrap.c
        lib_funcs/hpoly.c
        lib_funcs/fourier.c        
        lib_funcs/constelm.c
        lib_funcs/kernels.c
        lib_funcs/monitoring.c
        lib_funcs/piecewisepoly.c
        lib_funcs/pivoting.c
        lib_funcs/polynomials.c
        lib_funcs/space.c
        lib_linalg/linalg.c
        lib_linalg/matrix_util.c
        lib_optimization/optimization.c
        lib_quadrature/legquadrules2.c
        lib_quadrature/legquadrules.c
        lib_quadrature/quadrature.c
        lib_stringmanip/stringmanip.c
        lib_tensdecomp/candecomp.c
        lib_tensdecomp/convert.c
        lib_tensdecomp/cross.c
        lib_tensdecomp/cross.h
        lib_tensdecomp/tensortrain.c
        lib_tensdecomp/tt_integrate.c
        lib_tensdecomp/tt_multilinalg.c
        lib_tensor/tensor.c
        lib_probability/probability.c
        )



#create a libray containing everything!
file(GLOB SRC_FILES */*.c)
if (${static_libs} )
    add_library(c3 ${SRC_FILES})
  else ()
    if (MAKE_PYTHON_WRAPPERS)
      include_directories(${PYTHON_INCLUDE_DIRS})
      add_library(c3 SHARED ${SRC_FILES})
      target_link_libraries(c3
        ${blas_lapack}
        ${python_lib}
        -lm
        )
    else()
      add_library(c3 SHARED ${SRC_FILES})
      target_link_libraries(c3
        ${blas_lapack}
        -lm
        )
    endif()
 endif()


# configure_file("config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config_impl.h")
# add_library(c3 SHARED ${header} ${src})
# target_link_libraries(c3
#   ${blas_lapack}
#   -lm
#   )

target_include_directories(c3 PUBLIC
  $<BUILD_INTERFACE:${C3_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${include_dest}>
  $<INSTALL_INTERFACE:${lib_dest}>)

# install(TARGETS c3 EXPORT c3 DESTINATION "${lib_dest}")
# install(FILES ${header} DESTINATION "${include_dest}")

# install(TARGETS c3 EXPORT c3 DESTINATION "${lib_dest}")
install(TARGETS c3 DESTINATION lib)
install(FILES ${header} DESTINATION include/c3)

