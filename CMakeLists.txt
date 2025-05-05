cmake_minimum_required(VERSION 3.10)
project(Parallel-High-Pass-Filter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR C:/Users/Laila/Downloads/opencv/build)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# MPI Implementation
# add_executable(mpi_implementation src/mpi/mpi_implementation.cpp)
# target_link_libraries(mpi_implementation ${OpenCV_LIBS})

# # OpenMP Implementation
# add_executable(openmp_implementation src/openmp/openmp_implementation.cpp)
# target_link_libraries(openmp_implementation ${OpenCV_LIBS})

# Sequential Implementation
add_executable(sequential_implementation src/sequential/sequential_implementation.cpp)
target_link_libraries(sequential_implementation ${OpenCV_LIBS})
