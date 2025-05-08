# CSE455 - High Performance Computing: Parallel Programming Exercises

This repository contains solutions for the course **CSE455: High Performance Computing**, which focuses on implementing various computational problems using **parallel programming paradigms** including **OpenMP**, **MPI**, and **Sequential (baseline)** implementations.

## 🎯 Objective

The main goal of this repository is to demonstrate parallel thinking and programming skills by solving a variety of computational problems using:
- ✅ OpenMP (Shared-memory parallelism)
- ✅ MPI (Distributed-memory parallelism)
- ✅ Sequential Implementation (for comparison)

Each problem is benchmarked using intel VTune and compared in terms of:
- Execution time
- Scalability with respect to input size
- Scalability with respect to number of nodes/processes (for MPI)

## 📄 Project Description

This coursework requires:

- Handling **dynamic input sizes** and **dynamic number of nodes/processes**
- Testing on **diverse datasets and input sizes**
- Applying parallelism **as the main performance goal**
- For some image-based tasks (e.g., Low Pass Filter), datasets and images are provided via a shared drive

Team of **5 members** are required to:
- Implement each task using:
  1. **OpenMP**
  2. **MPI**
  3. **Sequential**



### 🧪 Deliverables (Included in this repo)
- 📁 Source Code for each implementation
- 📝 Report (in progress) containing:
  - Screenshots of input/output
  - Descriptions of the implementations
  - Performance comparisons and conclusions
- 📷 Visuals: Image outputs for tasks involving filters or image processing

## 📌 Bonus Objective (Extra Credit)
> Implement a **dynamic kernel filter size** instead of a fixed 3x3 size (for filter-based tasks).

## 🗂 Repository Structure
```
.
├── src/
│    ├── openmp/
│    │ ├── openmp_implementation.cpp
│    │ └── ...
│    ├── mpi/
│    │ ├── mpi_implementation.cpp
│    │ └── ...
│    ├── sequential/
│    │ ├── sequential_implementation.cpp
│    │ └── ...
│    └── common_utilities.cpp
├── images/
│ └── (input images / data samples used in testing)
├── report/
│ ├── report.pdf
│ └── screenshots/
└── README.md
```

## 🗂 How to run
### MPI
```
mpiexec -n 5 build\Debug\mpi_opencv_app.exe 3 images\input\lena.png
```