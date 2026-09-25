cmake_minimum_required(VERSION 3.15)

project(shrec2025_protein)

find_package(PCL 1.14 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Eigen3 CONFIG REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenMP)

add_executable(vtk2feat.bin vtk2feat.cpp sobol.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(vtk2feat.bin PRIVATE OpenMP::OpenMP_CXX ${PCL_LIBRARIES})
else()
  target_link_libraries(vtk2feat.bin ${PCL_LIBRARIES})
endif()
