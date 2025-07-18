# Author: Namcheol Lee
# Affiliation: Real-Time Operating System Laboratory, Seoul National University
# Contact: nclee@redwood.snu.ac.kr
# Date: 2025-07-02
# Description: Script for building SoftPreempt components

# libOpenCL_shim.so
g++ -shared -w -fPIC -O2 -I./common ./OpenCL_Shim/libOpenCL_shim.cpp -ldl -pthread -o libOpenCL_shim.so

# libOpenCL_profiling_shim.so
g++ -shared -w -fPIC -O2 -I./common ./OpenCL_Shim/libOpenCL_profiling_shim.cpp -ldl -pthread -o libOpenCL_profiling_shim.so

# OpenCL Kernel Scheduler
mkdir -p bin
g++ -o ./bin/opencl_kernel_scheduler -I./common ./OpenCL_Kernel_Scheduler/kernel_scheduler.cpp -ldl -lpthread
