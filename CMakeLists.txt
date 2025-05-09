cmake_minimum_required(VERSION 3.10)
project(high_pass_filter_app)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Set explicit MS-MPI paths (verify these match your installation)
set(MSMPI_ROOT "C:/Program Files (x86)/Microsoft SDKs/MPI")
set(MSMPI_INC "${MSMPI_ROOT}/Include")
set(MSMPI_LIB "${MSMPI_ROOT}/Lib/x64")

# Manual MPI configuration
set(MPI_CXX_INCLUDE_PATH "${MSMPI_INC}")
set(MPI_CXX_LIBRARIES "${MSMPI_LIB}/msmpi.lib")

find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_PATH}
)

# add_executable(high_pass_filter_app src/mpi/mpi_implementation.cpp)
# add_executable(high_pass_filter_app src/sequential/sequential_implementation.cpp)
add_executable(high_pass_filter_app src/openmp/openmp_implementation.cpp)

target_compile_features(high_pass_filter_app PRIVATE cxx_std_17)

target_link_libraries(high_pass_filter_app
    ${OpenCV_LIBS}
    ${MPI_CXX_LIBRARIES}
    "${MSMPI_LIB}/msmpi.lib"  # Explicit path
    "${MSMPI_LIB}/msmpifec.lib"  # Often required for MS-MPI
)