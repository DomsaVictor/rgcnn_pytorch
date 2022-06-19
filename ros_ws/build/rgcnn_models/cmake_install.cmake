# Install script for directory: /home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/src/rgcnn_models

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
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

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rgcnn_models" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/rgcnn_models/voxel_filter_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rgcnn_models" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/rgcnn_models/passthrough_filter_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rgcnn_models" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/rgcnn_models/plane_segmentation_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rgcnn_models" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/rgcnn_models/compute_volume_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rgcnn_models" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/rgcnn_models/compute_surface_normals_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages/rgcnn_models" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/lib/python2.7/dist-packages/rgcnn_models/__init__.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/lib/python2.7/dist-packages/rgcnn_models/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages/rgcnn_models" TYPE DIRECTORY FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/lib/python2.7/dist-packages/rgcnn_models/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/build/rgcnn_models/catkin_generated/installspace/rgcnn_models.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/rgcnn_models/cmake" TYPE FILE FILES
    "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/build/rgcnn_models/catkin_generated/installspace/rgcnn_modelsConfig.cmake"
    "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/build/rgcnn_models/catkin_generated/installspace/rgcnn_modelsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/rgcnn_models" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/src/rgcnn_models/package.xml")
endif()

