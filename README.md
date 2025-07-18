# SoftPreempt

SoftPreempt is a research project focused on software-level GPU preemption that does not require any kind of modification to hardware, system software (kernel), and user code. It intercepts OpenCL API calls from the user process and schedules it based on process priority for GPU preemption.

## Features

- **OpenCL Kernel Scheduler**: A user-space scheduler for OpenCL kernels, supporting priority queues and advanced scheduling algorithms.
- **OpenCL Shim**: Library (`libOpenCL_shim.so`) for intercepting OpenCL API calls of user processes and forwarding them to the OpenCL kernel scheduler.

## Directory Structure

```
build.sh    # Build script
common/     # Header files for both the OpenCL shim and kernel scheduler
    image_utils.hpp
    ipc.hpp
    logging.hpp
OpenCL_Kernel_Scheduler/ 
    kernel_scheduler.cpp # Main code of the OpenCL kernel scheduler
    include/             # Header files used by the OpenCL kernel scheduler
        enums_and_structs.hpp
        priority_kernel_queue.hpp
        CL/
OpenCL_Shim/            
    libOpenCL_profiling_shim.cpp # OpenCL shim for profiling purpose
    libOpenCL_shim.cpp   # OpenCL shim for intercepting API calls from user processes
```

## Building

To build the project and all components, run:

```sh
./build.sh
```

The resulting binaries and libraries will be placed in the `bin/` directory and the project root, respectively.

## Usage

### Applying the OpenCL Shim

To use the OpenCL shim library, follow these steps:

1. **Rename the Vendor Library**  
   Rename your existing vendor OpenCL library (typically in `/usr/lib`) to `libOpenCL_vendor.so`:
   ```sh
   sudo mv /usr/lib/libOpenCL.so /usr/lib/libOpenCL_vendor.so
   ```

2. **Install the Shim Library**  
   Copy `libOpenCL_shim.so` into `/usr/lib`:
   ```sh
   sudo cp ./libOpenCL_shim.so /usr/lib/
   ```

3. **Create a Symbolic Link**  
   Create a symbolic link named `libOpenCL.so` pointing to `libOpenCL_shim.so`:
   ```sh
   sudo ln -sf /usr/lib/libOpenCL_shim.so /usr/lib/libOpenCL.so
   ```

Now, all OpenCL applications on the system will use the shim library transparently.

> **Note:** To use `libOpenCL_profiling_shim.so`, follow the same steps above but with `libOpenCL_shim.so` as `libOpenCL_profiling_shim.so`

> **Note:** You can revert these changes by `ldconfig`. 

### Launching the OpenCL Kernel Scheduler

Run the kernel scheduler binary from `bin/opencl_kernel_scheduler`.

### Running applications

Now, you can run an application that uses LiteRT v1.4.0. We currently support only LiteRT. 

Or you can make up your own application that directly uses OpenCL and test it.

To assign priority to applications, you should run it with:
   ```sh
   sudo chrt -f [priority_level] [your_executable]
   ```

## Contact

For questions or collaboration, please contact the project maintainers.