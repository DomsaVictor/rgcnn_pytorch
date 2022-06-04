# Install script for directory: /home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/src/pointcloud_preparation

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

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pointcloud_preparation" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/pointcloud_preparation/voxel_filter_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pointcloud_preparation" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/pointcloud_preparation/passthrough_filter_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pointcloud_preparation" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/pointcloud_preparation/plane_segmentation_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pointcloud_preparation" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/pointcloud_preparation/compute_volume_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pointcloud_preparation" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/include/pointcloud_preparation/compute_surface_normals_nodeConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages/pointcloud_preparation" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/lib/python2.7/dist-packages/pointcloud_preparation/__init__.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/lib/python2.7/dist-packages/pointcloud_preparation/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages/pointcloud_preparation" TYPE DIRECTORY FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/devel/lib/python2.7/dist-packages/pointcloud_preparation/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/build/pointcloud_preparation/catkin_generated/installspace/pointcloud_preparation.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pointcloud_preparation/cmake" TYPE FILE FILES
    "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/build/pointcloud_preparation/catkin_generated/installspace/pointcloud_preparationConfig.cmake"
    "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/build/pointcloud_preparation/catkin_generated/installspace/pointcloud_preparationConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pointcloud_preparation" TYPE FILE FILES "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/ros_ws/src/pointcloud_preparation/package.xml")
endif()

