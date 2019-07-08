# Install script for directory: /Users/alex/Software/c3/c3

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/alex/Software/c3/build/lib.macosx-10.6-x86_64-3.6/libc3.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libc3.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libc3.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/alex/anaconda3/envs/idp/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libc3.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -add_rpath "/usr/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libc3.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -add_rpath "/Users/alex/anaconda3/envs/idp/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libc3.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libc3.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/c3" TYPE FILE FILES
    "/Users/alex/Software/c3/c3/lib_stringmanip/stringmanip.h"
    "/Users/alex/Software/c3/c3/lib_array/array.h"
    "/Users/alex/Software/c3/c3/lib_superlearn/learning_options.h"
    "/Users/alex/Software/c3/c3/lib_superlearn/objective_functions.h"
    "/Users/alex/Software/c3/c3/lib_superlearn/superlearn_util.h"
    "/Users/alex/Software/c3/c3/lib_superlearn/superlearn.h"
    "/Users/alex/Software/c3/c3/lib_superlearn/parameterization.h"
    "/Users/alex/Software/c3/c3/lib_superlearn/regress.h"
    "/Users/alex/Software/c3/c3/lib_interface/c3_interface.h"
    "/Users/alex/Software/c3/c3/lib_interface/approximate.h"
    "/Users/alex/Software/c3/c3/lib_clinalg/diffusion.h"
    "/Users/alex/Software/c3/c3/lib_clinalg/dmrg.h"
    "/Users/alex/Software/c3/c3/lib_clinalg/dmrgprod.h"
    "/Users/alex/Software/c3/c3/lib_clinalg/ft.h"
    "/Users/alex/Software/c3/c3/lib_clinalg/indmanage.h"
    "/Users/alex/Software/c3/c3/lib_clinalg/qmarray.h"
    "/Users/alex/Software/c3/c3/lib_clinalg/quasimatrix.h"
    "/Users/alex/Software/c3/c3/lib_clinalg/lib_clinalg.h"
    "/Users/alex/Software/c3/c3/lib_fft/fft.h"
    "/Users/alex/Software/c3/c3/lib_funcs/fapprox.h"
    "/Users/alex/Software/c3/c3/lib_funcs/functions.h"
    "/Users/alex/Software/c3/c3/lib_funcs/fwrap.h"
    "/Users/alex/Software/c3/c3/lib_funcs/hpoly.h"
    "/Users/alex/Software/c3/c3/lib_funcs/fourier.h"
    "/Users/alex/Software/c3/c3/lib_funcs/legtens.h"
    "/Users/alex/Software/c3/c3/lib_funcs/lib_funcs.h"
    "/Users/alex/Software/c3/c3/lib_funcs/linelm.h"
    "/Users/alex/Software/c3/c3/lib_funcs/constelm.h"
    "/Users/alex/Software/c3/c3/lib_funcs/kernels.h"
    "/Users/alex/Software/c3/c3/lib_funcs/monitoring.h"
    "/Users/alex/Software/c3/c3/lib_funcs/piecewisepoly.h"
    "/Users/alex/Software/c3/c3/lib_funcs/pivoting.h"
    "/Users/alex/Software/c3/c3/lib_funcs/polynomials.h"
    "/Users/alex/Software/c3/c3/lib_funcs/space.h"
    "/Users/alex/Software/c3/c3/lib_linalg/lib_linalg.h"
    "/Users/alex/Software/c3/c3/lib_linalg/linalg.h"
    "/Users/alex/Software/c3/c3/lib_linalg/matrix_util.h"
    "/Users/alex/Software/c3/c3/lib_optimization/lib_optimization.h"
    "/Users/alex/Software/c3/c3/lib_optimization/optimization.h"
    "/Users/alex/Software/c3/c3/lib_quadrature/legquadrules.h"
    "/Users/alex/Software/c3/c3/lib_quadrature/lib_quadrature.h"
    "/Users/alex/Software/c3/c3/lib_quadrature/quadrature.h"
    "/Users/alex/Software/c3/c3/lib_tensdecomp/candecomp.h"
    "/Users/alex/Software/c3/c3/lib_tensdecomp/cross.h"
    "/Users/alex/Software/c3/c3/lib_tensdecomp/tensortrain.h"
    "/Users/alex/Software/c3/c3/lib_tensdecomp/tt_integrate.h"
    "/Users/alex/Software/c3/c3/lib_tensdecomp/tt_multilinalg.h"
    "/Users/alex/Software/c3/c3/lib_tensor/tensor.h"
    "/Users/alex/Software/c3/c3/lib_probability/lib_probability.h"
    "/Users/alex/Software/c3/c3/lib_probability/probability.h"
    "/Users/alex/Software/c3/c3/lib_probability/probability_utils.h"
    )
endif()

