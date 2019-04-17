cmake_minimum_required (VERSION 3.0)
project (C3)

#create my librray
set(C3_VERSION_MAJOR 1 CACHE STRING "major version" FORCE)
set(C3_VERSION_MINOR 0 CACHE STRING "minor version" FORCE)
set(C3_VERSION "${C3_VERSION_MAJOR}.${C3_VERSION_MINOR}" CACHE STRING "version" FORCE)

option(C3_DEBUG_MODE "whether or not debug mode is activated" OFF)
option(MAKE_PYTHON_WRAPPERS "Create python wrappers" OFF)

include(c3.cmake)
c3_always_full_rpath()

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)

    option(USE_CLANG "build application with clang" ON)
    set(compiler_flags "-std=c99 -O3 -Wall -Weverything -Wno-shorten-64-to-32 -Wno-padded -Wno-missing-prototypes -Wno-format-nonliteral -Wno-disabled-macro-expansion -Wno-missing-noreturn -Wno-cast-qual -Wno-documentation-unknown-command ")
    set(CMAKE_C_COMPILER "clang")

    include_directories(/opt/local/include)
    # find_library(macblas
    #     NAMES ACCELERATE
    #     PATHS ${CMAKE_OSX_SYSROOT}/System/Library
    #     PATH_SUFFIXES Frameworks
    #     NO_DEFAULT_PATH)
    # # find_library(macblas
    # #     NAMES ACCELERATE
    # #     PATHS /System/Library
    # #     PATH_SUFFIXES Frameworks
    # #     NO_DEFAULT_PATH)  
    # set(blas_lapack ${macblas}/Accelerate)
    
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    include_directories(${BLAS_INCLUDE_DIR}, ${LAPACK_INCLUDE_DIR})
    set(blas_lapack ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})    
        


    #set(compiler_flags "-g -Wall -Wextra -O2 -std=c99 -flax-vector-conversions")
    #set(CMAKE_C_COMPILER "/usr/local/bin/gcc-6")
else ()

    include_directories(/usr/include)
  
    set(compiler_flags "-g -Wall -Wextra -O2 -std=c99 -fno-omit-frame-pointer")
    set(CMAKE_C_COMPILER "cc")

    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    include_directories(${BLAS_INCLUDE_DIR}, ${LAPACK_INCLUDE_DIR})
    set(blas_lapack ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

endif()

set(CMAKE_CXX_COMPILER "gcc")

if (MAKE_PYTHON_WRAPPERS)
  find_package(PythonInterp REQUIRED)
  set(__numpy_out 1)

  if (PYTHON_EXECUTABLE)
    # Find out the include path
    execute_process(
       COMMAND "${PYTHON_EXECUTABLE}" -c
      "from __future__ import print_function\ntry: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
       OUTPUT_VARIABLE __numpy_path)
    # And the version
    execute_process(
       COMMAND "${PYTHON_EXECUTABLE}" -c
       "from __future__ import print_function\ntry: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
       OUTPUT_VARIABLE __numpy_version)
  elseif(__numpy_out)
    message(STATUS "Python executable not found.")
  endif(PYTHON_EXECUTABLE)

  find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
    HINTS "${__numpy_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

  if(PYTHON_NUMPY_INCLUDE_DIR)
    set(PYTHON_NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
  endif(PYTHON_NUMPY_INCLUDE_DIR)

  include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

  # Rest of Python
  find_package(PythonLibs REQUIRED)
  set(python_lib ${PYTHON_LIBRARIES})

  # Add some more compiler flags
  # set(compiler_flags "${compiler_flags} -Wunused-function -Wno-reserved-id-macro -Wno-unused-function -Wno-duplicate-enum -Wno-unknown-warning-option")
  set(compiler_flags "${compiler_flags} -Wunused-function -Wno-unused-function")
endif()

option(BUILD_STATIC_LIB "Build a static library" OFF)
if (BUILD_STATIC_LIB)
  set(static_libs 1)  
else ()
  set(static_libs 0)
  set(compiler_flags "${compiler_flags} -fPIC")
endif()


option(BUILD_SUB_LIBS "Build all sub-libearies into libraries" OFF)

set(CMAKE_C_FLAGS "${compiler_flags}")

# SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(src)
include_directories(src/lib_stringmanip)
include_directories(src/lib_array)
include_directories(src/lib_linalg)
include_directories(src/lib_fft)
include_directories(src/lib_clinalg)
include_directories(src/lib_superlearn)
include_directories(src/lib_interface)
include_directories(src/lib_quadrature)
include_directories(src/lib_funcs)
include_directories(src/lib_tensor)
include_directories(src/lib_tensdecomp)
include_directories(src/lib_funcdecomp)
include_directories(src/lib_optimization)
include_directories(src/lib_probability)
include_directories(include)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(benchmarks)
add_subdirectory(profiling)
add_subdirectory(utilities)


# SWIGGING-BASED Wrappers
if (MAKE_PYTHON_WRAPPERS)
  add_subdirectory(wrappers/python)
endif()

