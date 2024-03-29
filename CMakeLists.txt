cmake_minimum_required (VERSION 3.10)
project (C3 C)

#create my librray
set(C3_VERSION_MAJOR 1 CACHE STRING "major version" FORCE)
set(C3_VERSION_MINOR 0 CACHE STRING "minor version" FORCE)
set(C3_VERSION "${C3_VERSION_MAJOR}.${C3_VERSION_MINOR}" CACHE STRING "version" FORCE)

option(C3_DEBUG_MODE "whether or not debug mode is activated" OFF)

include(c3.cmake)
c3_always_full_rpath()

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)

    option(USE_CLANG "build application with clang" ON)
    set(compiler_flags "-std=c99 -O3 -Wall -Wextra -Wno-shorten-64-to-32 -Wno-padded -Wno-missing-prototypes -Wno-format-nonliteral -Wno-disabled-macro-expansion -Wno-missing-noreturn -Wno-cast-qual -Wno-documentation-unknown-command") # "-Wnoreserved-id-macro" "-Wno-undef")
    # set(compiler_flags "-std=c99 -O3 -Wall -Wextra -Wno-shorten-64-to-32 -Wno-padded -Wno-missing-prototypes -Wno-format-nonliteral -Wno-disabled-macro-expansion -Wno-missing-noreturn -Wno-cast-qual -Wno-documentation-unknown-command -fvisibility=hidden") # "-Wnoreserved-id-macro" "-Wno-undef")
    
    set(CMAKE_C_COMPILER "clang")

    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    include_directories(${BLAS_INCLUDE_DIR}, ${LAPACK_INCLUDE_DIR})
    set(blas_lapack ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})    
        
else ()

    include_directories(/usr/include)
  
    set(compiler_flags "-g -Wall -Wextra -O2 -std=c99 -fno-omit-frame-pointer")
    set(CMAKE_C_COMPILER "cc")

    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    include_directories(${BLAS_INCLUDE_DIR}, ${LAPACK_INCLUDE_DIR})
    set(blas_lapack ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

endif()

option(BUILD_STATIC_LIB "Build a static library" OFF)
if (BUILD_STATIC_LIB)
  set(static_libs 1)
  set(compiler_flags "${compiler_flags} -fPIC")
else ()
  set(static_libs 0)
  set(compiler_flags "${compiler_flags} -fPIC")
endif()

option(BUILD_SUB_LIBS "Build all sub-libearies into libraries" OFF)
set(CMAKE_C_FLAGS "${compiler_flags}")

option(LIB_VISIBILITY_OFF "Make symbols invisible beyond local context" OFF)
if (LIB_VISIBILITY_OFF)
  add_compile_options(-fvisibility=hidden)
endif()

# SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(extra_dirs_c "c3_cex")
set(src_dirs "c3")

include_directories(c3)
include_directories(c3/lib_stringmanip)
include_directories(c3/lib_array)
include_directories(c3/lib_linalg)
include_directories(c3/lib_fft)
include_directories(c3/lib_clinalg)
include_directories(c3/lib_superlearn)
include_directories(c3/lib_interface)
include_directories(c3/lib_quadrature)
include_directories(c3/lib_funcs)
include_directories(c3/lib_tensor)
include_directories(c3/lib_tensdecomp)
include_directories(c3/lib_funcdecomp)
include_directories(c3/lib_optimization)
include_directories(c3/lib_probability)
include_directories(include)

add_subdirectory(c3)

option(BUILD_TESTS "Build Unit Tests" OFF)
option(BUILD_EXAMPLES "Build Examples" OFF)
option(BUILD_PROFILING "Build Profiling Tests" OFF)
option(BUILD_BENCHMARKS "Build Benchmarks Tests" OFF)
option(BUILD_UTILITIES "Build Utilities" OFF)
if (BUILD_TESTS)
  add_subdirectory(${extra_dirs_c}/test)
endif()
if (BUILD_EXAMPLES)
  add_subdirectory(${extra_dirs_c}/examples)
endif()
if (BUILD_PROFILING)
  add_subdirectory(${extra_dirs_c}/profiling)
endif()
if (BUILD_BENCHMARKS)
  add_subdirectory(${extra_dirs_c}/benchmarks)
endif()
if (BUILD_UTILITIES)
  add_subdirectory(${extra_dirs_c}/utilities)
endif()




