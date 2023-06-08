cmake_minimum_required(VERSION 3.8)
project(cuda_collision_checking)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

#if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
#endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(fcl REQUIRED)
find_package(CUDA REQUIRED)
find_package(assimp REQUIRED)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G" -lineinfo")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} )

add_library(cuda_collision_checking src/cuda_collision_checking.cu)
set_target_properties(cuda_collision_checking PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_compile_options(cuda_collision_checking PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)
target_link_libraries(cuda_collision_checking ${ros2-tiny-cuda-nn_LIBRARIES})

target_include_directories(cuda_collision_checking PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda_collision_checking PUBLIC ${ros2-tiny-cuda-nn_INCLUDE_DIR})
target_include_directories(cuda_collision_checking PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(TARGETS cuda_collision_checking DESTINATION lib/${PROJECT_NAME})


add_executable(main src/main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)
target_link_libraries(main ${ros2-tiny-cuda-nn_LIBRARIES})
target_link_libraries(main assimp::assimp)
target_link_libraries(main cuda_collision_checking)

target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(main PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(main PUBLIC ${ros2-tiny-cuda-nn_INCLUDE_DIR})
ament_target_dependencies(main rclcpp sensor_msgs moveit_msgs visualization_msgs)
target_include_directories(main PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

#install(TARGETS main DESTINATION lib/${PROJECT_NAME})


install(
    DIRECTORY data
    DESTINATION share/${PROJECT_NAME}
)


ament_package()


# debug
function(dump_cmake_variables)
  get_cmake_property(_variableNames VARIABLES)
  list (SORT _variableNames)
  foreach (_variableName ${_variableNames})
    if (ARGV0)
      unset(MATCHED)
      string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
      if (NOT MATCHED)
        continue()
      endif()
    endif()
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

#dump_cmake_variables()