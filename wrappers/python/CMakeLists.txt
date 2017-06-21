cmake_minimum_required (VERSION 3.0.2)

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

