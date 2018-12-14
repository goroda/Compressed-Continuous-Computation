cmake_minimum_required (VERSION 3.0)

add_custom_command(
  OUTPUT py_build_msg.log
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run.sh ${PYTHON_VERSION_MAJOR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  )

add_custom_target(
  PyWrapper ALL
  DEPENDS c3 py_build_msg.log
  )

set_target_properties(PyWrapper PROPERTIES EXCLUDE_FROM_ALL TRUE)


FIND_PACKAGE(SWIG REQUIRED)
include(${SWIG_USE_FILE})
include_directories(${PYTHON_INCLUDE_DIRS})
# swig_add_module(c3 python c3.i)

swig_add_library(c3 LANGUAGE python SOURCES c3.i)
swig_link_libraries(c3 c3 ${python_lib})

if (APPLE) 

   # interesting reading: https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/

   # Use rpath for _c3.so
   SET_TARGET_PROPERTIES(_c3 PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   SET_TARGET_PROPERTIES(_c3 PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where the c3 library is installed relative to where _simple_module.so is installed
   SET_TARGET_PROPERTIES(_c3 PROPERTIES INSTALL_RPATH "@loader_path/../../..")

else()
    SET_TARGET_PROPERTIES(_c3 PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    SET_TARGET_PROPERTIES(_c3 PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

endif()
       
