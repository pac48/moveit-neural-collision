cmake_minimum_required(VERSION 3.8)
project(ros2-tiny-cuda-nn)

# find dependencies
find_package(ament_cmake REQUIRED)

include(ExternalProject)
ExternalProject_Add(
    tiny-cuda-nn
    GIT_REPOSITORY https://github.com/pac48/tiny-cuda-nn.git
    GIT_TAG moveit2-collision
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
)


#install(DIRECTORY tmp DESTINATION include/)

ament_package(CONFIG_EXTRAS "tiny-cuda-nn-extras.cmake")


# debug
get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)

foreach(_variableName ${_variableNames})
  message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

