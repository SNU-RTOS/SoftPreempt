/********************************************************
 * Author: Namcheol Lee
 * Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * Contact: nclee@redwood.snu.ac.kr
 * Date: 2025-07-02
 * Description: OpenCL Shim of SoftPreempt
 ********************************************************/

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <linux/memfd.h>
#include <queue>
#include <sched.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include "image_utils.hpp"
#include "ipc.hpp"
#include "logging.hpp"

/* === Global Variables === */

// To reduce the number of IPC, we batch the kernel enqueue requests
constexpr size_t MAX_KERNEL_BATCH_SIZE = 8;
std::unordered_map<cl_command_queue, std::vector<KernelEnqueueInfo>> kernel_requests_in_batch; // command queue to kernel request info vector map
std::unordered_map<cl_kernel, std::vector<KernelArg>> arg_map; // kernel to argument vector map, for passing kernel arguments

typedef void* FuncPtr;
struct FunctionEntry {
    const char* name;
    FuncPtr pointer;
};

static const FunctionEntry scheduler_function_table[] = {
    {"clGetPlatformIDs", (FuncPtr)clGetPlatformIDs},
    {"clGetPlatformInfo", (FuncPtr)clGetPlatformInfo},
    {"clGetDeviceIDs", (FuncPtr)clGetDeviceIDs},
    {"clGetDeviceInfo", (FuncPtr)clGetDeviceInfo},
    {"clCreateContext", (FuncPtr)clCreateContext},
    {"clCreateProgramWithSource", (FuncPtr)clCreateProgramWithSource},
    {"clBuildProgram", (FuncPtr)clBuildProgram},
    {"clCreateKernel", (FuncPtr)clCreateKernel},
    {"clCreateBuffer", (FuncPtr)clCreateBuffer},
    {"clSetKernelArg", (FuncPtr)clSetKernelArg},
    {"clCreateCommandQueue", (FuncPtr)clCreateCommandQueue},
    {"clCreateCommandQueueWithProperties", (FuncPtr)clCreateCommandQueueWithProperties},
    {"clEnqueueWriteBuffer", (FuncPtr)clEnqueueWriteBuffer},
    {"clEnqueueNDRangeKernel", (FuncPtr)clEnqueueNDRangeKernel},
    {"clEnqueueReadBuffer", (FuncPtr)clEnqueueReadBuffer},
    {"clGetSupportedImageFormats", (FuncPtr)clGetSupportedImageFormats},
    {"clGetKernelWorkGroupInfo", (FuncPtr)clGetKernelWorkGroupInfo},
    {"clCreateImage", (FuncPtr)clCreateImage},
    {"clCreateSubBuffer", (FuncPtr)clCreateSubBuffer},
    {"clReleaseMemObject", (FuncPtr)clReleaseMemObject},
    {"clReleaseKernel", (FuncPtr)clReleaseKernel},
    {"clReleaseProgram", (FuncPtr)clReleaseProgram},
    {"clReleaseCommandQueue", (FuncPtr)clReleaseCommandQueue},
    {"clReleaseContext", (FuncPtr)clReleaseContext},
    {"clRetainProgram", (FuncPtr)clRetainProgram},
    {"clFlush", (FuncPtr)clFlush},
    {"clFinish", (FuncPtr)clFinish},
    {nullptr, nullptr} // end marker
};

// Priority
/* 
* The theoretically priority range SoftPreempt supports is between -1 and 100
* However, for implementation ease, we set the priority range between 0 and 255
* 0 is the lowest priority, for best-effort processes
* 1 to 100 is for processes that are applied with SCHED_FIFO, SCHED_RR, or other real-time scheduling policy of Linux
* 101 to 255 is reserved
*/
uint32_t priority = 0; // lowest priority as default

/* === End of Global Variables === */

/* === Inline Functions === */

// Creates an anonymous in-memory file (memfd), resizes it, and maps it into memory
static inline int create_memfd(const char* tag, size_t sz, void** file) {
    // Create an anonymous memory-backed file with close-on-exec flag
    int fd = syscall(SYS_memfd_create, tag, MFD_CLOEXEC);

    // Resize the in-memory file to the desired size
    ftruncate(fd, sz);

    // Map the file (file) into memory with read/write permissions
    *file = mmap(nullptr, sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

    return fd;
}

// Send a file descriptor (fd) and a header struct (hdr) over a UNIX domain socket (sock)
static inline void send_fd_with_hdr(int sock, int fd, const ShmHeader& hdr) {
    // Prepare the header
    struct iovec iov{ const_cast<ShmHeader*>(&hdr), sizeof(hdr) };

    // Allocate buffer for the control message (enough for 1 file descriptor)
    char ctrl[CMSG_SPACE(sizeof(fd))] = {};

    // Set up the message header for sendmsg
    msghdr msg{};
    msg.msg_iov = &iov;                  // Points to data buffer
    msg.msg_iovlen = 1;                  // One data buffer
    msg.msg_control = ctrl;              // Points to control message buffer
    msg.msg_controllen = sizeof(ctrl);   // Size of control message buffer

    // Prepare the control message to send the file descriptor
    cmsghdr* c = CMSG_FIRSTHDR(&msg);    // Get pointer to first control message
    c->cmsg_level = SOL_SOCKET;          // Indicates socket-level control message
    c->cmsg_type  = SCM_RIGHTS;          // We're sending a file descriptor
    c->cmsg_len   = CMSG_LEN(sizeof(fd)); // Length of this control message

    // Copy the file descriptor into the control message payload
    memcpy(CMSG_DATA(c), &fd, sizeof(fd));

    // Send the message with both the header and file descriptor
    sendmsg(sock, &msg, 0);
}

// Connects to the OpenCL kernel scheduler via UNIX domain socket
// Returns the socket file descriptor on success, or CL_OUT_OF_HOST_MEMORY on failure
static inline int connect_to_kernel_scheduler_int() {
    // Create a UNIX domain socket
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket connection error");
        return CL_OUT_OF_HOST_MEMORY;
    }

    // Set up the socket address structure
    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    // Attempt to connect to the server socket
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(sock);
        return CL_OUT_OF_HOST_MEMORY;
    }

    return sock;
}

// Connects to the OpenCL kernel scheduler via UNIX domain socket
// Returns the socket file descriptor on success, or nullptr on failure
static inline int* connect_to_kernel_scheduler_ptr() {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket connection error");
        return nullptr;
    }

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(sock);
        return nullptr;
    }

    return new int(sock);
}

/* === End of Inline Functions === */

/* === Helper Functions === */

// Helper function to send batch of kernel execution requests
cl_int send_kernel_execution_requests(cl_command_queue queue) {
    auto it = kernel_requests_in_batch.find(queue);
    if (it == kernel_requests_in_batch.end() || it->second.empty()) {
        return CL_SUCCESS;
    }

    const auto& batch = it->second;
    size_t total_kernels = batch.size();

    // === Step 1: Calculate total size ===
    size_t total_size = 0;
    std::vector<std::vector<KernelArg>> all_args;
    all_args.reserve(total_kernels);

    for (auto& info : batch) {
        cl_kernel kernel = reinterpret_cast<cl_kernel>(info.kernel_rid);
        const auto& args = arg_map[kernel];

        all_args.push_back(args);  // for later copy
        total_size += sizeof(KernelEnqueueInfo);
        total_size += args.size() * sizeof(KernelArg);
    }

    // === Step 2: Allocate shared memory ===
    void* shm_ptr = nullptr;
    int shm_fd = create_memfd("ocl_kernel_batch", total_size, &shm_ptr);
    char* ptr = reinterpret_cast<char*>(shm_ptr);

    for (size_t i = 0; i < total_kernels; ++i) {
        const KernelEnqueueInfo& orig = batch[i];
        KernelEnqueueInfo copy = orig;
        copy.num_args = all_args[i].size();

        memcpy(ptr, &copy, sizeof(KernelEnqueueInfo));
        ptr += sizeof(KernelEnqueueInfo);

        memcpy(ptr, all_args[i].data(), copy.num_args * sizeof(KernelArg));
        ptr += copy.num_args * sizeof(KernelArg);
    }

    // === Step 3: Connect and send ===
    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_ENQUEUE_NDRANGE_KERNEL;
    write(sock, &req_type, sizeof(req_type));

    ShmHeader hdr{SHM_TYPE_KERNEL, 0, static_cast<uint32_t>(total_size), static_cast<uint32_t>(total_kernels)};
    send_fd_with_hdr(sock, shm_fd, hdr);

    munmap(shm_ptr, total_size);
    close(shm_fd);

    cl_int status;
    read(sock, &status, sizeof(status));
    close(sock);

    // === Step 4: Cleanup ===
    kernel_requests_in_batch[queue].clear();

    return status;
}

/* === End of Helper Functions */

/* === OpenCL Wrappers === */
/*
* Flow when an OpenCL wrapper is called
* 1. Make socket connection to the kernel scheduler (Some wrappers may not need this)
* 2. Send request type and parameters
* 3. Send shared memory file descriptor if needed
* 4. Receive response from the kernel scheduler
*/
// Note that only the 29 OpenCL functions that are used by LiteRT v1.4.0 are implemented
extern "C" {   // <-- C linkage to avoid function name mangling of the g++ compiler

// GCC extension
// The function is called when the library is loaded
__attribute__((constructor))
void on_shim_load() {
    LOG_INFO("[SHIM] libOpenCL_shim.so loaded!\n");
}

void* clGetExtensionFunctionAddress(const char* func_name) {
    if (!func_name) {
        LOG_ERROR("[SHIM] clGetExtensionFunctionAddress called with NULL func_name");
        return nullptr;
    }

    LOG_INFO("[SHIM] clGetExtensionFunctionAddress called for: " << func_name);

    for (const FunctionEntry* entry = scheduler_function_table; entry->name != nullptr; ++entry) {
        if (strcmp(entry->name, func_name) == 0) {
            LOG_INFO("[SHIM] Found function: " << func_name << " → " << entry->pointer);
            return entry->pointer;
        }
    }

    LOG_ERROR("[SHIM] Function not found: " << func_name);
    return nullptr;
}

void* clGetExtensionFunctionAddressForPlatform(cl_platform_id platform, const char* func_name) {
    LOG_INFO("[SHIM] clGetExtensionFunctionAddressForPlatform called for: " << func_name);
    return clGetExtensionFunctionAddress(func_name);
}

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_GET_PLATFORM_IDS;
    write(sock, &req_type, sizeof(req_type));

    cl_int status;
    cl_uint count;
    read(sock, &status, sizeof(status));
    read(sock, &count, sizeof(count));
    
    if (num_platforms) *num_platforms = count;
    if (platforms) platforms[0] = reinterpret_cast<cl_platform_id>(PLATFORM_RESOURCE_ID);

    close(sock);
    return status;
}

cl_int clGetPlatformInfo(cl_platform_id platform,
                         cl_platform_info param_name,
                         size_t param_value_size,
                         void* param_value,
                         size_t* param_value_size_ret) {
    if (platform != reinterpret_cast<cl_platform_id>(PLATFORM_RESOURCE_ID)) return CL_INVALID_PLATFORM;

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_GET_PLATFORM_INFO;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &param_name, sizeof(param_name));
    write(sock, &param_value_size, sizeof(param_value_size));

    uint32_t has_param_value = (param_value != nullptr);
    write(sock, &has_param_value, sizeof(has_param_value));

    cl_int status;
    size_t actual_size = 0;
    read(sock, &status, sizeof(status));
    read(sock, &actual_size, sizeof(actual_size));

    if (param_value && actual_size > 0) {
        read(sock, param_value, actual_size);
    }

    if (param_value_size_ret) *param_value_size_ret = actual_size;

    close(sock);
    return status;
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
    cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) {
    if (platform != reinterpret_cast<cl_platform_id>(PLATFORM_RESOURCE_ID)) return CL_INVALID_PLATFORM;

    if (num_entries == 0 && devices != nullptr) {
        return CL_INVALID_VALUE;
    }

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_GET_DEVICE_IDS;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &device_type, sizeof(device_type));
    write(sock, &num_entries, sizeof(num_entries));

    cl_int status;
    cl_uint count;
    read(sock, &status, sizeof(status));
    read(sock, &count, sizeof(count));

    if (num_devices) *num_devices = count;
    if (devices && count > 0) devices[0] = reinterpret_cast<cl_device_id>(DEVICE_RESOURCE_ID);

    close(sock);
    return status;
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    if (device != reinterpret_cast<cl_device_id>(DEVICE_RESOURCE_ID)) {
        LOG_ERROR("[SHIM] Invalid device resource id!");
        return CL_INVALID_DEVICE;
    }

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_GET_DEVICE_INFO;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &param_name, sizeof(param_name));
    write(sock, &param_value_size, sizeof(param_value_size));

    cl_int status;
    size_t actual_size = 0;
    read(sock, &status, sizeof(status));
    read(sock, &actual_size, sizeof(actual_size));

    if (param_value && actual_size > 0) {
        read(sock, param_value, actual_size);
    }

    if (param_value_size_ret) *param_value_size_ret = actual_size;

    close(sock);
    return status;
}

cl_context clCreateContext(const cl_context_properties *properties,
    cl_uint num_devices,
    const cl_device_id *devices,
    void (*pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) {
    
    int* sock_ptr = connect_to_kernel_scheduler_ptr();
    int sock;
    if (!sock_ptr) {
        return nullptr;
    } else {
        sock = *sock_ptr;
        delete sock_ptr;  // Clean up the pointer after use
    }

    // Retreiving scheduling policy and priority
    int policy;
    struct sched_param param;

    policy = sched_getscheduler(0);  // 0 means current process
    if (policy == -1) {
        perror("sched_getscheduler");
        exit(1);
    }

    if (sched_getparam(0, &param) == -1) {
        priority = 0;
    } else {
        const char *policy_name =
            (policy == SCHED_FIFO) ? "SCHED_FIFO" :
            (policy == SCHED_RR)   ? "SCHED_RR" :
            (policy == SCHED_OTHER)? "SCHED_OTHER" :
            "UNKNOWN";

        printf("Scheduling policy: %s\n", policy_name);

        if(policy == SCHED_OTHER){
            priority = 0; // SCHED_OTHER
        } else if (policy == SCHED_FIFO || policy == SCHED_RR) {
            priority = param.sched_priority + 1;
        }
        printf("Priority: %d\n", priority);
    }
    

    uint32_t req_type = REQ_CREATE_CONTEXT;
    pid_t pid = getpid();
    write(sock, &req_type, sizeof(req_type));
    write(sock, &pid, sizeof(pid));
    write(sock, &num_devices, sizeof(num_devices));
    write(sock, &devices[0], sizeof(devices[0])); // only one device for now
    write(sock, &priority, sizeof(priority)); 

    cl_int status;
    ResourceId context_rid = reinterpret_cast<uint64_t>(CONTEXT_RESOURCE_ID);
    read(sock, &status, sizeof(status));

    close(sock);

    if (errcode_ret) *errcode_ret = status;
    return (cl_context)context_rid;
}

cl_program clCreateProgramWithSource(cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret) {

    int* sock_ptr = connect_to_kernel_scheduler_ptr();
    int sock;
    if (!sock_ptr) {
        return nullptr;
    } else {
        sock = *sock_ptr;
        delete sock_ptr;  // Clean up the pointer after use
    }


    uint32_t req_type = REQ_CREATE_PROGRAM_WITH_SOURCE;
    write(sock, &req_type, sizeof(req_type));

    write(sock, &count, sizeof(count));

    size_t total_size = 0;
    for (cl_uint i = 0; i < count; ++i)
        total_size += lengths ? lengths[i] : strlen(strings[i]);

    void* map;
    int fd = create_memfd("ocl_prog_src", total_size, &map);

    size_t offset = 0;
    for (cl_uint i = 0; i < count; ++i) {
        size_t len = lengths ? lengths[i] : strlen(strings[i]);
        memcpy((char*)map + offset, strings[i], len);
        offset += len;
    }

    ShmHeader hdr{SHM_TYPE_BUFFER, 0, total_size, count};
    send_fd_with_hdr(sock, fd, hdr);

    munmap(map, total_size);
    close(fd);

    cl_int status;
    ResourceId program_rid;
    read(sock, &status, sizeof(status));
    read(sock, &program_rid, sizeof(program_rid));

    close(sock);

    if (errcode_ret) *errcode_ret = status;
    return (cl_program)program_rid;
}

cl_int clBuildProgram(cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (*pfn_notify)(cl_program, void*),
    void* user_data) {

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_BUILD_PROGRAM;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &program, sizeof(program));

    uint32_t has_options = (options != nullptr);
    write(sock, &has_options, sizeof(has_options));
    if (has_options) {
        uint32_t options_len = strlen(options);
        write(sock, &options_len, sizeof(options_len));
        write(sock, options, options_len);
    }

    cl_int status;
    read(sock, &status, sizeof(status));

    close(sock);
    return status;
}

cl_kernel clCreateKernel(cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret) {
    
    int* sock_ptr = connect_to_kernel_scheduler_ptr();
    int sock;
    if (!sock_ptr) {
        return nullptr;
    } else {
        sock = *sock_ptr;
        delete sock_ptr;  // Clean up the pointer after use
    }



    uint32_t req_type = REQ_CREATE_KERNEL;
    cl_uint name_len = std::strlen(kernel_name);

    write(sock, &req_type, sizeof(req_type));
    write(sock, &program, sizeof(program));
    write(sock, &name_len, sizeof(name_len));
    write(sock, kernel_name, name_len);

    cl_int status;
    ResourceId kernel_rid;
    read(sock, &status, sizeof(status));
    read(sock, &kernel_rid, sizeof(kernel_rid));

    close(sock);
    if (errcode_ret) *errcode_ret = status;
    return (cl_kernel)kernel_rid;
}

cl_mem clCreateBuffer(cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret) {

    int* sock_ptr = connect_to_kernel_scheduler_ptr();
    int sock;
    if (!sock_ptr) {
        return nullptr;
    } else {
        sock = *sock_ptr;
        delete sock_ptr;  // Clean up the pointer after use
    }



    // decide whether we need shared memory
    bool need_shm = (host_ptr && (flags & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)));
    uint32_t req = REQ_CREATE_BUFFER;
    write(sock, &req, sizeof(req));          // existing request id

    write(sock, &flags, sizeof(flags));
    write(sock, &size, sizeof(size));
    
    if(!need_shm) {
        /* Allocates only buffer space, just send metadata */
        uint32_t no_shm = 0;
        write(sock, &no_shm, sizeof(no_shm));
    } else {
        uint32_t no_shm = 1;
        write(sock, &no_shm, sizeof(no_shm));
        // 1. create memfd and copy data if COPY_HOST_PTR
        void* map; int fd = create_memfd("ocl_buf", size, &map);
        if (flags & CL_MEM_COPY_HOST_PTR) memcpy(map, host_ptr, size);

        // 2. send metadata + FD
        ShmHeader hdr{SHM_TYPE_BUFFER, flags, size, 0};
        send_fd_with_hdr(sock, fd, hdr);

        // 3. we can drop our map immediately (OpenCL has its own copy now)
        munmap(map, size); close(fd);
    }

    cl_int status;
    ResourceId buffer_rid;
    read(sock, &status, sizeof(status));
    read(sock, &buffer_rid, sizeof(buffer_rid));

    close(sock);
    if (errcode_ret) *errcode_ret = status;
    
    return (cl_mem)buffer_rid;
}

cl_int clSetKernelArg(cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value) {
    KernelArg arg;
    arg.index = arg_index;
    arg.size = arg_size;
    
    if (arg_size > sizeof(arg.value)) {
        LOG_ERROR("[SHIM] clSetKernelArg: arg_size too large " << arg_size);
        return CL_INVALID_VALUE;
    }

    if (arg_value && arg_size > 0) {
        memcpy(arg.value, arg_value, arg_size);
    } else {
        memset(arg.value, 0, sizeof(arg.value));
    }

    auto& vec = arg_map[kernel];

    // Replace existing argument or append
    bool found = false;
    for (auto& existing : vec) {
        if (existing.index == arg_index) {
            existing = arg;
            found = true;
            break;
        }
    }
    if (!found) vec.push_back(arg);

    return CL_SUCCESS;
}

// We do not allow processes to create a command queue
// This will return the ResourceId of a non-existing command queue
cl_command_queue clCreateCommandQueueWithProperties(cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret) {

    int* sock_ptr = connect_to_kernel_scheduler_ptr();
    int sock;
    if (!sock_ptr) {
        return nullptr;
    } else {
        sock = *sock_ptr;
        delete sock_ptr;  // Clean up the pointer after use
    }

    uint32_t req_type = REQ_CREATE_COMMAND_QUEUE_WITH_PROPERTIES;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &context, sizeof(context));
    write(sock, &device, sizeof(device));

    uint32_t has_properties = (properties != nullptr);
    write(sock, &has_properties, sizeof(has_properties));

    if (has_properties) {
        std::vector<cl_queue_properties> props_vec;
        for (const cl_queue_properties* p = properties; p && *p != 0; p += 2) {
            props_vec.push_back(p[0]);
            props_vec.push_back(p[1]);
        }
        props_vec.push_back(0); // terminator

        uint32_t props_count = props_vec.size();
        write(sock, &props_count, sizeof(props_count));
        write(sock, props_vec.data(), sizeof(cl_queue_properties) * props_count);
    }

    cl_int status;
    ResourceId cmd_queue_rid;
    read(sock, &status, sizeof(status));
    read(sock, &cmd_queue_rid, sizeof(cmd_queue_rid));

    close(sock);
    if (errcode_ret) *errcode_ret = status;
    return (cl_command_queue)cmd_queue_rid;
}

// This function is deprecated since OpenCL 2.0
// It is now redirected to clCreateCommandQueueWithProperties
cl_command_queue clCreateCommandQueue(cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret) {

    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, (cl_queue_properties)properties,
        0
    };
    return clCreateCommandQueueWithProperties(context, device, props, errcode_ret);
}

// Events are not supported since DL inference runtimes do not usually use them
cl_int clEnqueueWriteBuffer(cl_command_queue queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events,
    const cl_event* event_wait_list,
    cl_event* event) {
    
    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_ENQUEUE_WRITE_BUFFER;
    write(sock, &req_type, sizeof(req_type));
    pid_t pid = getpid();
    write(sock, &pid, sizeof(pid));
    write(sock, &queue, sizeof(queue));
    write(sock, &buffer, sizeof(buffer));
    write(sock, &blocking_write, sizeof(blocking_write));
    write(sock, &offset, sizeof(offset));
    write(sock, &size, sizeof(size));
    
    void* map;
    int fd = create_memfd("ocl_write", size, &map);
    memcpy(map, ptr, size);

    ShmHeader hdr{SHM_TYPE_BUFFER, 0, size, 0};
    send_fd_with_hdr(sock, fd, hdr);

    munmap(map, size);
    close(fd);

    cl_int status;
    read(sock, &status, sizeof(status));

    close(sock);
    return status;
}

cl_int clEnqueueNDRangeKernel(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events,
    const cl_event* event_wait_list,
    cl_event* event) {
    
    KernelEnqueueInfo info;
    info.pid = getpid();
    info.cmd_queue_rid = reinterpret_cast<ResourceId>(queue);
    info.kernel_rid = reinterpret_cast<ResourceId>(kernel);
    info.work_dim = work_dim;

    size_t zero[3] = {0, 0, 0};
    memcpy(info.global_offset, global_work_offset ?: zero, sizeof(size_t) * 3);
    memcpy(info.global_work_size, global_work_size ?: zero, sizeof(size_t) * 3);
    memcpy(info.local_work_size, local_work_size ?: zero, sizeof(size_t) * 3);
    // We defer setting info.num_args until flush time (from arg_map)

    auto& batch = kernel_requests_in_batch[queue];
    batch.push_back(std::move(info));

    if (batch.size() >= MAX_KERNEL_BATCH_SIZE) {
        return send_kernel_execution_requests(queue);
    }

    return CL_SUCCESS;
}

cl_int clEnqueueReadBuffer(cl_command_queue queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void* ptr,
    cl_uint num_events,
    const cl_event* event_wait_list,
    cl_event* event) {

    cl_int send_request_status = send_kernel_execution_requests(queue);

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_ENQUEUE_READ_BUFFER;
    write(sock, &req_type, sizeof(req_type));
    pid_t pid = getpid();
    write(sock, &pid, sizeof(pid));
    write(sock, &queue, sizeof(queue));
    write(sock, &buffer, sizeof(buffer));
    write(sock, &blocking_read, sizeof(blocking_read));
    write(sock, &offset, sizeof(offset));
    write(sock, &size, sizeof(size));

    void* map;
    int fd = create_memfd("ocl_read", size, &map);
    memset(map, 0xA5, size);  // pattern-fill for debugging

    ShmHeader hdr{SHM_TYPE_BUFFER, 0, size, 0};
    send_fd_with_hdr(sock, fd, hdr);
    
    munmap(map, size);
    // close(fd); // for debugging, we can check if the fd is closed properly

    cl_int status = CL_SUCCESS;
    read(sock, &status, sizeof(status));

    if (status == CL_SUCCESS) {
        // Re-open and copy back, not required since we did not close fd
        // fd = memfd_create("ocl_read_reopen", MFD_CLOEXEC);
        // ftruncate(fd, size);
        map = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);

        memcpy(ptr, map, size);
        munmap(map, size);
        close(fd);
    }

    close(sock);
    
    return (send_request_status != CL_SUCCESS) ? send_request_status : status;
}

cl_int clGetSupportedImageFormats(cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats) {

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_GET_SUPPORTED_IMAGE_FORMATS;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &flags, sizeof(flags));
    write(sock, &image_type, sizeof(image_type));
    write(sock, &num_entries, sizeof(num_entries));

    cl_int status;
    cl_uint count;
    read(sock, &status, sizeof(status));
    read(sock, &count, sizeof(count));

    if (num_image_formats) *num_image_formats = count;

    if (count > 0) {
        std::vector<cl_image_format> tmp(count);
        read(sock, tmp.data(), sizeof(cl_image_format) * count);

        if (image_formats && num_entries > 0) {
            size_t copy_count = std::min((size_t)count, (size_t)num_entries);
            memcpy(image_formats, tmp.data(), sizeof(cl_image_format) * copy_count);
        }
    }

    close(sock);
    return status;
}

cl_int clGetKernelWorkGroupInfo(cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) {

    if (device != reinterpret_cast<cl_device_id>(DEVICE_RESOURCE_ID)) return CL_INVALID_DEVICE;

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_GET_KERNEL_WORK_GROUP_INFO;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &kernel, sizeof(kernel));
    write(sock, &param_name, sizeof(param_name));
    write(sock, &param_value_size, sizeof(param_value_size));

    cl_int status;
    size_t actual_size = 0;
    read(sock, &status, sizeof(status));
    read(sock, &actual_size, sizeof(actual_size));

    if (param_value && actual_size > 0) {
        read(sock, param_value, actual_size);
    }

    if (param_value_size_ret) *param_value_size_ret = actual_size;

    close(sock);
    return status;
}

cl_mem clCreateImage(cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret) {
    
    int* sock_ptr = connect_to_kernel_scheduler_ptr();
    int sock;
    if (!sock_ptr) {
        return nullptr;
    } else {
        sock = *sock_ptr;
        delete sock_ptr;  // Clean up the pointer after use
    }



    uint32_t req_type = REQ_CREATE_IMAGE;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &flags, sizeof(flags));

    // format
    uint32_t has_format = (image_format != nullptr);
    write(sock, &has_format, sizeof(has_format));
    if (has_format) {
        write(sock, image_format, sizeof(cl_image_format));
    }

    // desc
    uint32_t has_desc = (image_desc != nullptr);
    write(sock, &has_desc, sizeof(has_desc));
    if (has_desc) {
        write(sock, image_desc, sizeof(cl_image_desc));
    }

    // host_ptr
    bool need_shm = (host_ptr && (flags & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)));
    if (!need_shm) {
        uint32_t no_shm = 0;
        write(sock, &no_shm, sizeof(no_shm));
    } else {
        uint32_t no_shm = 1;
        write(sock, &no_shm, sizeof(no_shm));
        size_t host_size = calculate_image_size(image_format, image_desc);
        void* map; int fd = create_memfd("ocl_img", host_size, &map);
        if (flags & CL_MEM_COPY_HOST_PTR) memcpy(map, host_ptr, host_size);
    
        ShmHeader hdr{SHM_TYPE_IMAGE, flags, host_size, 0};
        send_fd_with_hdr(sock, fd, hdr);
        munmap(map, host_size);
        close(fd);
    }

    cl_int status;
    ResourceId image_rid;
    read(sock, &status, sizeof(status));
    read(sock, &image_rid, sizeof(image_rid));

    close(sock);
    if (errcode_ret) { *errcode_ret = status; } 

    return (cl_mem)image_rid;
}

cl_mem clCreateSubBuffer(cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type create_type,
    const void* create_info,
    cl_int* errcode_ret) {

    int* sock_ptr = connect_to_kernel_scheduler_ptr();
    int sock;
    if (!sock_ptr) {
        return nullptr;
    } else {
        sock = *sock_ptr;
        delete sock_ptr;  // Clean up the pointer after use
    }



    uint32_t req_type = REQ_CREATE_SUBBUFFER;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &buffer, sizeof(buffer));
    write(sock, &flags, sizeof(flags));
    write(sock, &create_type, sizeof(create_type));
    write(sock, create_info, sizeof(cl_buffer_region));

    cl_int status;
    ResourceId sub_buffer_rid;
    read(sock, &status, sizeof(status));
    read(sock, &sub_buffer_rid, sizeof(sub_buffer_rid));

    close(sock);
    if (errcode_ret) *errcode_ret = status;
    return (cl_mem)sub_buffer_rid;
}

cl_int clReleaseMemObject(cl_mem memobj) {
    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_RELEASE_MEM_OBJECT;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &memobj, sizeof(memobj));

    cl_int status;
    read(sock, &status, sizeof(status));
    close(sock);
    return status;
}

cl_int clReleaseKernel(cl_kernel kernel) {
    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_RELEASE_KERNEL;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &kernel, sizeof(kernel));

    cl_int status;
    read(sock, &status, sizeof(status));
    close(sock);
    return status;
}

cl_int clReleaseProgram(cl_program program) {
    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_RELEASE_PROGRAM;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &program, sizeof(program));

    cl_int status;
    read(sock, &status, sizeof(status));
    close(sock);
    return status;
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_RELEASE_COMMAND_QUEUE;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &command_queue, sizeof(command_queue));

    cl_int status;
    read(sock, &status, sizeof(status));
    close(sock);
    return status;
}

cl_int clReleaseContext(cl_context context) {
    cl_int status = CL_SUCCESS;

    return status;
}

cl_int clRetainProgram(cl_program program) {
    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_RETAIN_PROGRAM;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &program, sizeof(program));

    cl_int status;
    read(sock, &status, sizeof(status));
    close(sock);
    return status;
}

cl_int clFlush(cl_command_queue command_queue) {

    cl_int send_request_status = send_kernel_execution_requests(command_queue);

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_FLUSH;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &command_queue, sizeof(command_queue));

    cl_int status;
    read(sock, &status, sizeof(status));
    close(sock);

    return (send_request_status != CL_SUCCESS) ? send_request_status : status;
}

cl_int clFinish(cl_command_queue command_queue) {

    cl_int send_request_status = send_kernel_execution_requests(command_queue);

    int sock = connect_to_kernel_scheduler_int();

    uint32_t req_type = REQ_FINISH;
    write(sock, &req_type, sizeof(req_type));
    write(sock, &command_queue, sizeof(command_queue));

    cl_int status;
    read(sock, &status, sizeof(status));
    close(sock);

    return (send_request_status != CL_SUCCESS) ? send_request_status : status;
}

} /* === End of Extern "C" === */  

/* === END OF FILE === */