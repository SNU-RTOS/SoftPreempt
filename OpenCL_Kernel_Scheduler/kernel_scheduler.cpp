/********************************************************
 * Author: Namcheol Lee
 * Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * Contact: nclee@redwood.snu.ac.kr
 * Date: 2025-07-02
 * Description: OpenCL Kernel Scheduler of SoftPreempt
 ********************************************************/

#define CL_TARGET_OPENCL_VERSION 300
#include <array>
#include <atomic>
#include <cstring>
#include <condition_variable>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <signal.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "include/CL/cl.h"
#include "include/CL/cl_ext.h"
#include "include/priority_kernel_queue.hpp"
#include "ipc.hpp"
#include "image_utils.hpp"
#include "include/enums_and_structs.hpp"
#include "logging.hpp"

/* === Variable Naming Conventions === */
/*
* Variables with _cl_ prefix are the variables created through the vendor OpenCL library
* variables without _cl_ prefix are the variables of OpenCL Kernel Scheduler
*/

/* === Global Variables === */
// System-wide vendor OpenCL library pointer
void* vendor_opencl_lib = nullptr;

// System-wide OpenCL objects: platform, device, context, and command queue
cl_platform_id _cl_platform = nullptr;
cl_device_id _cl_device = nullptr;
cl_context _cl_context = nullptr;
cl_command_queue _cl_command_queue = nullptr;

// System-wide ResourceIds for platform, device, and context
ResourceId platform_rid = 0;
ResourceId device_rid = 0;  
ResourceId context_rid = 0; 

// Global counters to guarantee unique ResourceId
static uint64_t program_rid_counter = 1;
static uint64_t kernel_rid_counter = 1;
static uint64_t mem_rid_counter = 1;
static uint64_t cmd_queue_rid_counter = 1;

// Maps for managing ResourceId and OpenCL objects
std::unordered_map<ResourceId, cl_program> program_rid_to_cl_map;
std::unordered_map<ResourceId, KernelInfo> kernel_rid_to_cl_map;
std::unordered_map<ResourceId, cl_mem> mem_rid_to_cl_map;
std::unordered_map<cl_kernel, cl_ulong> kernel_to_execution_time_map;
std::unordered_map<cl_kernel, uint32_t> kernel_to_execution_count_map;

// Variables related to priority arbitration
constexpr uint32_t PRIORITY_LEVELS = 256; // We support between 0 - 100 but we define 256 for implementation ease
std::unordered_map<pid_t, uint32_t> pid_to_priority_map; // Map from process ID to priority level
static std::atomic<uint64_t> top_bitmap = 0; // bitmap for the top-level priorities
static std::atomic<uint64_t> priority_bitmap[(PRIORITY_LEVELS + 63) / 64] = {}; // bitmap for the priority levels
static std::array<PriorityKernelQueue<KernelRequestInfo, 2048>, PRIORITY_LEVELS> priority_kernel_queue;
std::unordered_map<ResourceId, std::atomic<int>> remaining_kernel_request_per_proc; // For tracking the number of kernels in the priority kernel queue per process, maps command queue rid to the number of kernels

// Condition variables to handle clFinish calls from user process
std::unordered_map<ResourceId, std::condition_variable> clFinish_cvars;
std::unordered_map<ResourceId, std::mutex> clFinish_mutexes;

// eventfd for waking up the kernel scheduling thread
int wake_fd = -1;

/* === End of Global Variables === */

/* === Inline Functions === */

// Receives a file descriptor and a header of shared memory
static inline bool recv_fd_with_hdr(int sock, ShmHeader& hdr, int& out_fd) {
    char ctrl[CMSG_SPACE(sizeof(int))] = {};
    struct iovec iov{ &hdr, sizeof(hdr) };
    struct msghdr msg{};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = ctrl;
    msg.msg_controllen = sizeof(ctrl);

    ssize_t r = recvmsg(sock, &msg, MSG_CMSG_CLOEXEC);
    if (r != sizeof(hdr)) {
        perror("recvmsg failed");
        LOG_INFO("[SCHEDULER] recvmsg failed: " << r);
        return false;
    }

    int received_fd = -1;
    out_fd = -1;
    for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
            memcpy(&received_fd, CMSG_DATA(c), sizeof(received_fd));
            break;
        }
    }

    if (received_fd < 0) {
        LOG_ERROR("[recv_fd_with_hdr] Warning: FD not received!\n");
        return false;
    }

    // Duplicate the FD to ensure it remains valid after sender closes
    out_fd = fcntl(received_fd, F_DUPFD_CLOEXEC, 0);
    close(received_fd);  // close the original FD from recvmsg

    if (out_fd < 0) {
        LOG_ERROR("[recv_fd_with_hdr] fcntl(F_DUPFD_CLOEXEC) failed");
        return false;
    }

    return true;
}

// Set the priority bit in the bitmap
inline void set_priority_bit(uint32_t prio) {
    uint32_t idx = prio / 64;
    uint32_t bit = prio % 64;
    priority_bitmap[idx].fetch_or(1ULL << bit, std::memory_order_relaxed);
    top_bitmap.fetch_or(1ULL << idx, std::memory_order_relaxed);
}

// Clear the priority bit in the bitmap
inline void clear_priority_bit(uint32_t prio) {
    uint32_t idx = prio / 64;
    uint32_t bit = prio % 64;
    priority_bitmap[idx].fetch_and(~(1ULL << bit), std::memory_order_relaxed);

    if (priority_bitmap[idx].load(std::memory_order_relaxed) == 0) {
        top_bitmap.fetch_and(~(1ULL << idx), std::memory_order_relaxed);
    }
}

// Get the highest priority from the bitmap
inline uint32_t get_highest_priority() {
    uint64_t top = top_bitmap.load(std::memory_order_relaxed);
    if (top == 0) return PRIORITY_LEVELS;

    uint32_t idx = 63 - __builtin_clzll(top); // MSB
    uint64_t bits = priority_bitmap[idx].load(std::memory_order_relaxed);
    uint32_t bit = 63 - __builtin_clzll(bits); // MSB
    return idx * 64 + bit;
}

// Push a kernel request into the priority kernel queue
inline void push_kernel_request(uint32_t prio, cl_command_queue _cl_q, KernelRequestInfo&& kernel_request_info) {
    if (prio >= PRIORITY_LEVELS) prio = 0;

    bool was_empty = false;
    bool ok = priority_kernel_queue[prio].push(std::move(kernel_request_info), was_empty);
    if (!ok) {
        LOG_ERROR("[SCHED] PriorityKernelQueue full for prio " << prio);
        return;
    }

    // set bitmap bits exactly as before
    set_priority_bit(prio);

    // Wake dispatcher only if queue transitioned empty → non-empty
    if (was_empty) {
        uint64_t one = 1;
        LOG_INFO("[SCHEDULER] Waking up dispatcher");
        if (eventfd_write(wake_fd, one) < 0) {
            perror("eventfd_write");
        }
    }
}

// === End of Inline Functions ===

/* === Helper Functions === */

// Function for creating an integer identifier for internal management
ResourceId create_resource_id(ResourceType type, uint64_t local_id) {
    ResourceId rid = ((uint64_t)type << 56) | (local_id & 0x00FFFFFFFFFFFFFFULL);

    return rid;
}

// Function for using the functions defined in vendor-provided OpenCL library
void* load_vendor_func(const char* name) {
    // just for safety, check if the OpenCL library is loaded
    if (!vendor_opencl_lib) {
        vendor_opencl_lib = dlopen("/usr/lib/libOpenCL_vendor.so", RTLD_NOW | RTLD_LOCAL); // Change the path to the vendor-provided OpenCL library if needed
        if (!vendor_opencl_lib) {
            LOG_ERROR("[SCHEDULER] dlopen failed: " << dlerror());
            exit(1);
        }
    }
    void* sym = dlsym(vendor_opencl_lib, name);
    if (!sym) {
        LOG_ERROR("[SCHEDULER] dlsym failed for " << name << ": " << dlerror());
        exit(1);
    }
    return sym;
}

// Function for initializing the OpenCL kernel scheduler
void init_scheduler() {
    cl_int status;

    // Dynamically load the vendor-provided OpenCL library
    if (!vendor_opencl_lib) {
        vendor_opencl_lib = dlopen("/usr/lib/libOpenCL_vendor.so", RTLD_NOW | RTLD_LOCAL);
        if (!vendor_opencl_lib) {
            LOG_ERROR("[SCHEDULER] dlopen failed: " << dlerror());
            exit(1);
        }
    }

    // Get the platform of the device, we assume that there is only one platform
    auto clGetPlatformIDsFn = (cl_int (*)(cl_uint, cl_platform_id*, cl_uint*)) load_vendor_func("clGetPlatformIDs");

    cl_platform_id platforms[4] = {};
    cl_uint num_platforms = 0;
    status = clGetPlatformIDsFn(4, platforms, &num_platforms);

    if (status != CL_SUCCESS || num_platforms == 0) {
        LOG_ERROR("[SCHEDULER] Failed to get OpenCL platforms.");
        exit(1);
    }
    _cl_platform = platforms[0];

    // After getting the platform, we can get the device IDs, we assume that there is only one GPU in the platform
    auto clGetDeviceIDsFn = (cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*)) load_vendor_func("clGetDeviceIDs");

    cl_device_id devices[4] = {};
    cl_uint num_devices = 0;
    status = clGetDeviceIDsFn(_cl_platform, CL_DEVICE_TYPE_GPU, 4, devices, &num_devices);

    if (status != CL_SUCCESS || num_devices == 0) {
        LOG_ERROR("[SCHEDULER] Failed to get GPU devices.");
        exit(1);
    }
    _cl_device = devices[0];

    // Now we create a context for managing OpenCL resources, function calls in a centralized way
    auto clCreateContextFn = (cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*)) load_vendor_func("clCreateContext");

    _cl_context = clCreateContextFn(nullptr, 1, &_cl_device, nullptr, nullptr, &status);

    if (status != CL_SUCCESS || !_cl_context) {
        LOG_ERROR("[SCHEDULER] Failed to create OpenCL context.");
        exit(1);
    }

    // Create a single command queue
    auto clCreateCommandQueueWithPropertiesFn = (cl_command_queue (*)(cl_context, cl_device_id, const cl_queue_properties*, cl_int*)) load_vendor_func("clCreateCommandQueueWithProperties");
    const cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0  // Null-terminator
    };

    _cl_command_queue = clCreateCommandQueueWithPropertiesFn(_cl_context, _cl_device, props, &status);
    if (status != CL_SUCCESS) {
        LOG_ERROR("[SCHEDULER] Failed to create Kernel command queue.");
        exit(1);
    }

    // Defined in common/ipc.hpp
    platform_rid = PLATFORM_RESOURCE_ID; 
    device_rid = DEVICE_RESOURCE_ID; 
    context_rid = CONTEXT_RESOURCE_ID;

    LOG_INFO("[SCHEDULER] Scheduler initialized successfully.");
}
/* === End of Helper Functions === */

/* === Request Handling Functions === */
/*
* These functions are called by the request reception thread
* when a request is received from the user process.
*/
void handle_clGetPlatformIDs(int client_fd) {

    auto clGetPlatformIDsFn = (cl_int (*)(cl_uint, cl_platform_id*, cl_uint*)) load_vendor_func("clGetPlatformIDs");
    
    cl_platform_id platforms[4] = {};
    cl_uint num_platforms = 0;
    cl_int status = clGetPlatformIDsFn(4, platforms, &num_platforms);

    LOG_INFO("[SCHEDULER] clGetPlatformIDs → " << status << ", count = " << num_platforms);

    if (status == CL_SUCCESS && num_platforms > 0) {
        LOG_INFO("[SCHEDULER] _cl_platform = " << _cl_platform);
    }

    write(client_fd, &status, sizeof(status));
    write(client_fd, &num_platforms, sizeof(num_platforms));
}

void handle_clGetPlatformInfo(int client_fd) {
    cl_platform_info param_name;
    size_t param_value_size;

    read(client_fd, &param_name, sizeof(param_name));
    read(client_fd, &param_value_size, sizeof(param_value_size));

    uint32_t has_param_value = 0;
    read(client_fd, &has_param_value, sizeof(has_param_value));

    char buffer[512] = {};
    void* param_value_ptr = has_param_value ? buffer : nullptr;
    size_t actual_size = 0;

    auto clGetPlatformInfoFn = (cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*)) load_vendor_func("clGetPlatformInfo");

    cl_int status = clGetPlatformInfoFn(_cl_platform, param_name, param_value_size, param_value_ptr, &actual_size);

    write(client_fd, &status, sizeof(status));
    write(client_fd, &actual_size, sizeof(actual_size));

    if (status == CL_SUCCESS && has_param_value && actual_size > 0) {
        write(client_fd, buffer, actual_size);
    }
}

// We do not send the actual device ID to the user process
void handle_clGetDeviceIDs(int client_fd) {
    cl_device_type device_type;
    cl_uint num_entries;
    read(client_fd, &device_type, sizeof(device_type));
    read(client_fd, &num_entries, sizeof(num_entries));

    auto clGetDeviceIDsFn = (cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*)) load_vendor_func("clGetDeviceIDs");

    cl_device_id devices[4] = {};
    cl_uint count = 0;

    cl_int status = clGetDeviceIDsFn(_cl_platform, device_type, num_entries, (num_entries > 0 ? devices : nullptr), &count);

    write(client_fd, &status, sizeof(status));
    write(client_fd, &count, sizeof(count));
}

void handle_clGetDeviceInfo(int client_fd) {
    cl_device_info param_name;
    size_t param_value_size;
    read(client_fd, &param_name, sizeof(param_name));
    read(client_fd, &param_value_size, sizeof(param_value_size));

    auto clGetDeviceInfoFn = (cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*)) load_vendor_func("clGetDeviceInfo");

    char buffer[512] = {};
    size_t actual_size = 0;

    cl_int status = clGetDeviceInfoFn(_cl_device, param_name, sizeof(buffer), buffer, &actual_size);

    write(client_fd, &status, sizeof(status));
    write(client_fd, &actual_size, sizeof(actual_size));
    if (actual_size > 0) {
        write(client_fd, buffer, actual_size);
    }
}

void handle_clCreateContext(int client_fd) {
    pid_t pid;
    cl_uint num_devices;
    ResourceId _device_rid_handle;
    uint32_t priority = 256;
    read(client_fd, &pid, sizeof(pid));
    read(client_fd, &num_devices, sizeof(num_devices));
    read(client_fd, &_device_rid_handle, sizeof(_device_rid_handle));
    read(client_fd, &priority, sizeof(priority));

    pid_to_priority_map[pid] = priority;

    using _clCreateContextFn = cl_context (*)(const cl_context_properties*,
                                             cl_uint,
                                             const cl_device_id*,
                                             void (*pfn_notify)(const char*, const void*, size_t, void*),
                                             void*,
                                             cl_int*);

    auto clCreateContextFn = (_clCreateContextFn) load_vendor_func("clCreateContext");

    cl_int status = CL_SUCCESS;

    write(client_fd, &status, sizeof(status));
}

void handle_clCreateProgramWithSource(int client_fd) {
    cl_uint count;

    read(client_fd, &count, sizeof(count));

    ShmHeader hdr{};
    int shm_fd = -1;
    bool ok = recv_fd_with_hdr(client_fd, hdr, shm_fd);
    if (!ok || shm_fd < 0 || hdr.size == 0) {
        cl_int fail = CL_INVALID_VALUE;
        ResourceId dummy = 0;
        write(client_fd, &fail, sizeof(fail));
        write(client_fd, &dummy, sizeof(dummy));
        return;
    }

    void* src_map = mmap(nullptr, hdr.size, PROT_READ, MAP_SHARED, shm_fd, 0);

    std::vector<const char*> strings;
    std::vector<size_t> lengths;
    size_t chunk_size = hdr.size / count;
    for (cl_uint i = 0; i < count; ++i) {
        strings.push_back((const char*)src_map + i * chunk_size);
        lengths.push_back(chunk_size);
    }

    auto clCreateProgramWithSourceFn = (cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*)) load_vendor_func("clCreateProgramWithSource");

    cl_int status = CL_SUCCESS;
    cl_program _cl_program = clCreateProgramWithSourceFn(_cl_context, count, strings.data(), lengths.data(), &status);

    munmap(src_map, hdr.size);
    close(shm_fd);

    ResourceId program_rid = create_resource_id(RESOURCE_TYPE_PROGRAM, program_rid_counter++);

    if (status == CL_SUCCESS) {
        program_rid_to_cl_map[program_rid] = _cl_program;
    } else {
        program_rid = 0;
        LOG_ERROR("[SCHEDULER] clCreateProgramWithSource failed, status = " << status);
    }

    write(client_fd, &status, sizeof(status));
    write(client_fd, &program_rid, sizeof(program_rid));
}

void handle_clBuildProgram(int client_fd) {
    ResourceId program_rid;
    read(client_fd, &program_rid, sizeof(program_rid));

    uint32_t has_options = 0;
    read(client_fd, &has_options, sizeof(has_options));

    std::string options_str;
    if (has_options) {
        uint32_t options_len = 0;
        read(client_fd, &options_len, sizeof(options_len));
        options_str.resize(options_len);
        read(client_fd, options_str.data(), options_len);
    }

    cl_program _cl_program = nullptr;
    bool found = false;

    _cl_program = program_rid_to_cl_map[program_rid];
    if (_cl_program) {
        found = true;
    }

    cl_int status = CL_INVALID_PROGRAM;
    if (found) {
        auto clBuildProgramFn = (cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*)) load_vendor_func("clBuildProgram");

        const char* options_cstr = (has_options ? options_str.c_str() : nullptr);

        status = clBuildProgramFn(_cl_program, 1, &_cl_device, options_cstr, nullptr, nullptr);
    } else {
        LOG_ERROR("[SCHEDULER] Failed to locate program for build: " << program_rid);
    }

    write(client_fd, &status, sizeof(status));
}

void handle_clCreateKernel(int client_fd) {
    ResourceId program_rid;
    cl_uint name_len;
    read(client_fd, &program_rid, sizeof(program_rid));
    read(client_fd, &name_len, sizeof(name_len));

    std::vector<char> name_buf(name_len + 1, '\0');
    read(client_fd, name_buf.data(), name_len);
    std::string kernel_name(name_buf.data(), name_len);

    cl_program _cl_program = program_rid_to_cl_map[program_rid];

    cl_kernel _cl_kernel = nullptr;
    cl_int status = CL_INVALID_PROGRAM;
    if (_cl_program) {
        auto clCreateKernelFn = (cl_kernel (*)(cl_program, const char*, cl_int*)) load_vendor_func("clCreateKernel");

        _cl_kernel = clCreateKernelFn(_cl_program, kernel_name.c_str(), &status);

        if (status == CL_SUCCESS) {
            ResourceId kernel_rid = create_resource_id(RESOURCE_TYPE_KERNEL, kernel_rid_counter++);

            KernelInfo kinfo = {
                ._cl_kernel = _cl_kernel,
                .kernel_name = kernel_name,
            };
            kernel_rid_to_cl_map[kernel_rid] = kinfo;
            kernel_to_execution_time_map[_cl_kernel] = 0.0; // Initialize execution time
            kernel_to_execution_count_map[_cl_kernel] = 0; // Initialize execution count

            write(client_fd, &status, sizeof(status));
            write(client_fd, &kernel_rid, sizeof(kernel_rid));
            return;
        }
    }

    // Failure management
    LOG_ERROR("[SCHEDULER] Failed to create kernel '" << kernel_name << "'.");
    write(client_fd, &status, sizeof(status));
    ResourceId null_handle = 0;
    write(client_fd, &null_handle, sizeof(null_handle));
}

void handle_clCreateBuffer(int client_fd) {
    cl_mem_flags flags;
    size_t size;
    uint32_t no_shm;

    read(client_fd, &flags, sizeof(flags));
    read(client_fd, &size, sizeof(size));
    read(client_fd, &no_shm, sizeof(no_shm));

    ShmHeader hdr{}; int shm_fd = -1;
    bool got_shm;
    if (no_shm)
        got_shm = recv_fd_with_hdr(client_fd, hdr, shm_fd);   // might fail gracefully

    void* host_ptr = nullptr;
    std::vector<char> host_data;

    if (got_shm && shm_fd >= 0) {
        host_ptr = mmap(nullptr, hdr.size, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (!(flags & CL_MEM_USE_HOST_PTR)) {
            /* OpenCL copies immediately – we can unmap right after create */ 
        }
    } else {
        flags &= ~(CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR);
    }

    cl_int status = CL_INVALID_CONTEXT;
    cl_mem _cl_mem = nullptr;
    ResourceId buffer_rid = 0;

    auto clCreateBufferFn = (cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*)) load_vendor_func("clCreateBuffer");

    _cl_mem = clCreateBufferFn(_cl_context, flags, size, host_ptr, &status);

    if (status == CL_SUCCESS) {
        buffer_rid = create_resource_id(RESOURCE_TYPE_MEM_OBJECT, mem_rid_counter++);
        mem_rid_to_cl_map[buffer_rid] = _cl_mem;
    } else {
        LOG_ERROR("[SCHEDULER] clCreateBuffer failed, status = " << status);
    }

    write(client_fd, &status, sizeof(status));
    write(client_fd, &buffer_rid, sizeof(buffer_rid));
}

// Never called
void handle_clSetKernelArg(int client_fd) {}

void handle_clCreateCommandQueueWithProperties(int client_fd) {
    ResourceId context_rid;
    ResourceId device_rid;
    uint32_t has_properties;

    read(client_fd, &context_rid, sizeof(context_rid));
    read(client_fd, &device_rid, sizeof(device_rid));
    read(client_fd, &has_properties, sizeof(has_properties));

    std::vector<cl_queue_properties> props;
    if (has_properties) {
        uint32_t props_count = 0;
        read(client_fd, &props_count, sizeof(props_count));

        props.resize(props_count);
        if (props_count > 0) {
            read(client_fd, props.data(), sizeof(cl_queue_properties) * props_count);
        }
    } else {
        LOG_ERROR("[SCHEDULER] No properties received (has_properties == 0)");
    }

    /* 
    * We do not allow creating a command queue
    * We just fake the creation of a command queue
    * To handle cases where a user process creates multiple command queues
    * We do not ignore it on the OpenCL Shim side
    */
    cl_int status = CL_SUCCESS;
    ResourceId cmd_queue_rid = create_resource_id(RESOURCE_TYPE_QUEUE, cmd_queue_rid_counter++);
    cl_command_queue _cl_queue = nullptr;
    remaining_kernel_request_per_proc[cmd_queue_rid] = 0;

    write(client_fd, &status, sizeof(status));
    write(client_fd, &cmd_queue_rid, sizeof(cmd_queue_rid));
}

void handle_clEnqueueWriteBuffer(int client_fd) {
    ResourceId cmd_queue_rid, buffer_rid;
    cl_bool blocking_write;
    size_t offset, size;
    pid_t pid;
    read(client_fd, &pid, sizeof(pid));
    read(client_fd, &cmd_queue_rid, sizeof(cmd_queue_rid)); // not needed
    read(client_fd, &buffer_rid, sizeof(buffer_rid));
    read(client_fd, &blocking_write, sizeof(blocking_write));
    read(client_fd, &offset, sizeof(offset));
    read(client_fd, &size, sizeof(size));

    // Receive data through shared memory
    ShmHeader hdr{};
    int shm_fd = -1;
    bool got_shm = recv_fd_with_hdr(client_fd, hdr, shm_fd);

    if (!got_shm || shm_fd < 0) {
        LOG_ERROR("[SCHEDULER] Failed to receive shared memory for WriteBuffer");
        cl_int status = CL_INVALID_VALUE;
        write(client_fd, &status, sizeof(status));
        return;
    }
    
    // Create and enqueue the command
    uint32_t prio = (pid_to_priority_map.count(pid) ? pid_to_priority_map[pid] : 0);
    KernelRequestInfo kernel_request_info{};
    kernel_request_info.type = TYPE_WRITE_BUFFER;
    kernel_request_info.client_fd = client_fd;
    kernel_request_info.cmd_queue_rid = cmd_queue_rid;
    kernel_request_info.buffer = mem_rid_to_cl_map[buffer_rid];
    kernel_request_info.blocking = blocking_write;
    kernel_request_info.offset = offset;
    kernel_request_info.size = size;
    kernel_request_info.shm_fd = shm_fd;
    kernel_request_info.shm_size = hdr.size;

    push_kernel_request(prio, _cl_command_queue, std::move(kernel_request_info));
    remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid]++;

    cl_int status = CL_SUCCESS;
    write(client_fd, &status, sizeof(status));
}

void handle_clEnqueueNDRangeKernel(int client_fd) {
    // Function for setting kernel args
    auto clSetKernelArgFn = (cl_int (*)(cl_kernel, cl_uint, size_t, const void*))load_vendor_func("clSetKernelArg");

    cl_command_queue _cl_q = _cl_command_queue;
    if (!_cl_q) return;

    // Receive data through shared memory
    ShmHeader hdr;
    int shm_fd = -1;
    bool ok = recv_fd_with_hdr(client_fd, hdr, shm_fd);
    if (!ok || shm_fd < 0 || hdr.size < sizeof(KernelEnqueueInfo)) {
        LOG_ERROR("[SCHEDULER] Invalid or missing SHM for NDRangeKernel");
        cl_int status = CL_INVALID_VALUE;
        write(client_fd, &status, sizeof(status));
        return;
    }

    // Respond immediately to unblock client
    cl_int status = CL_SUCCESS;
    write(client_fd, &status, sizeof(status));

    void* map = mmap(nullptr, hdr.size, PROT_READ, MAP_SHARED, shm_fd, 0);
    char* ptr = reinterpret_cast<char*>(map);

    for (uint32_t i = 0; i < hdr.extra; ++i) {
        // Parse KernelEnqueueInfo ===
        KernelEnqueueInfo* info = reinterpret_cast<KernelEnqueueInfo*>(ptr);
        ptr += sizeof(KernelEnqueueInfo);

        cl_kernel _cl_kernel = kernel_rid_to_cl_map[info->kernel_rid]._cl_kernel;
        if (!_cl_kernel) {
            LOG_ERROR("[SCHEDULER] Kernel not found: " << info->kernel_rid);
            continue;
        }

        // Parse args and set them ===
        KernelArg* arg_list = reinterpret_cast<KernelArg*>(ptr);
        ptr += sizeof(KernelArg) * info->num_args;

        for (uint32_t j = 0; j < info->num_args; ++j) {
            auto& arg = arg_list[j];
            cl_int arg_status = CL_SUCCESS;

            // Create a safe local copy of the arg value
            unsigned char safe_copy[sizeof(arg.value)] = {};
            memcpy(safe_copy, arg.value, arg.size);

            if (arg.size == sizeof(ResourceId)) {
                ResourceId maybe = *reinterpret_cast<ResourceId*>(arg.value);
                auto it = mem_rid_to_cl_map.find(maybe);
                if (it != mem_rid_to_cl_map.end()) {
                    cl_mem _cl_mem = it->second;
                    arg_status = clSetKernelArgFn(_cl_kernel, arg.index, arg.size, &_cl_mem);
                } else {
                    arg_status = clSetKernelArgFn(_cl_kernel, arg.index, arg.size, safe_copy);
                }
            } else {
                arg_status = clSetKernelArgFn(_cl_kernel, arg.index, arg.size, safe_copy);
            }

            if (arg_status != CL_SUCCESS) {
                LOG_ERROR("[SCHEDULER] clSetKernelArg failed on arg " << arg.index
                          << ", status = " << arg_status);
            }
        }

        // Enqueue kernel into the priority kernel queue
        uint32_t prio = pid_to_priority_map.count(info->pid)
                        ? pid_to_priority_map[info->pid]
                        : 0;

        KernelRequestInfo kernel_request_info;
        kernel_request_info.type = TYPE_NDRANGE_KERNEL;
        kernel_request_info.cmd_queue_rid = info->cmd_queue_rid;
        kernel_request_info.work_dim = info->work_dim;
        kernel_request_info.global_offset = std::vector<size_t>(info->global_offset, info->global_offset + info->work_dim);
        kernel_request_info.global_size   = std::vector<size_t>(info->global_work_size, info->global_work_size + info->work_dim);
        kernel_request_info.local_size    = std::vector<size_t>(info->local_work_size, info->local_work_size + info->work_dim);
        kernel_request_info._cl_kernel = _cl_kernel;

        if (kernel_request_info._cl_kernel) {
            push_kernel_request(prio, _cl_q, std::move(kernel_request_info));
            remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid]++;
        }
    }

    munmap(map, hdr.size);
    close(shm_fd);
}

void handle_clEnqueueReadBuffer(int client_fd){
    ResourceId cmd_queue_rid, buffer_rid; 
    cl_bool blocking; 
    size_t offset, size;
    pid_t pid;
    read(client_fd,&pid,sizeof(pid));
    read(client_fd,&cmd_queue_rid,sizeof(cmd_queue_rid));
    read(client_fd,&buffer_rid,sizeof(buffer_rid));
    read(client_fd,&blocking,sizeof(blocking));
    read(client_fd,&offset,sizeof(offset));
    read(client_fd,&size,sizeof(size));

    cl_command_queue _cl_q = _cl_command_queue; 
    cl_mem _cl_buf = mem_rid_to_cl_map[buffer_rid];
    uint32_t prio = (pid_to_priority_map.count(pid) ? pid_to_priority_map[pid] : 0);

    if(!_cl_q||!_cl_buf) { 
        cl_int st=CL_INVALID_VALUE;
        write(client_fd,&st,sizeof(st));
        close(client_fd); return; 
    }

    // Receive data through shared memory
    ShmHeader hdr{}; int shm_fd = -1;
    bool got_shm = recv_fd_with_hdr(client_fd, hdr, shm_fd);

    KernelRequestInfo kernel_request_info;
    kernel_request_info.type=TYPE_READ_BUFFER;
    kernel_request_info.cmd_queue_rid = cmd_queue_rid;
    kernel_request_info.buffer=_cl_buf; kernel_request_info.offset=offset; kernel_request_info.size=size; 
    kernel_request_info.blocking=blocking; kernel_request_info.client_fd=client_fd;
    kernel_request_info.shm_fd = shm_fd;
    kernel_request_info.shm_size = hdr.size;

    push_kernel_request(prio, _cl_q, std::move(kernel_request_info));
    remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid]++;
}

void handle_clGetSupportedImageFormats(int client_fd) {
    cl_mem_flags flags;
    cl_mem_object_type image_type;
    cl_uint num_entries;

    read(client_fd, &flags, sizeof(flags));
    read(client_fd, &image_type, sizeof(image_type));
    read(client_fd, &num_entries, sizeof(num_entries));

    cl_int status = CL_INVALID_CONTEXT;

    if (_cl_context) {
        auto clGetSupportedImageFormatsFn = (cl_int (*)(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format*, cl_uint*)) load_vendor_func("clGetSupportedImageFormats");

        std::vector<cl_image_format> formats(num_entries);
        cl_uint count = 0;
        status = clGetSupportedImageFormatsFn(_cl_context, flags, image_type, num_entries, formats.data(), &count);

        write(client_fd, &status, sizeof(status));
        write(client_fd, &count, sizeof(count));

        if (status == CL_SUCCESS && count > 0) {
            write(client_fd, formats.data(), sizeof(cl_image_format) * count);
        }
        return;
    }

    // fallback: in case the _cl_context is not set
    LOG_ERROR("[SCHEDULER] Invalid context handle for clGetSupportedImageFormats");
    cl_uint zero = 0;
    write(client_fd, &status, sizeof(status));
    write(client_fd, &zero, sizeof(zero));
}

void handle_clGetKernelWorkGroupInfo(int client_fd) {
    ResourceId kernel_rid;
    cl_kernel_work_group_info param_name;
    size_t param_value_size;

    read(client_fd, &kernel_rid, sizeof(kernel_rid));
    read(client_fd, &param_name, sizeof(param_name));
    read(client_fd, &param_value_size, sizeof(param_value_size));

    cl_kernel _cl_kernel = kernel_rid_to_cl_map[kernel_rid]._cl_kernel;

    cl_int status = CL_INVALID_KERNEL;
    size_t actual_size = 0;
    char buffer[512] = {};

    if (_cl_kernel) {
        auto clGetKernelWorkGroupInfoFn = (cl_int (*)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*)) load_vendor_func("clGetKernelWorkGroupInfo");

        status = clGetKernelWorkGroupInfoFn(_cl_kernel, _cl_device, param_name, sizeof(buffer), buffer, &actual_size);

    } else {
        LOG_ERROR("[SCHEDULER] Invalid kernel rid for clGetKernelWorkGroupInfo");
    }

    write(client_fd, &status, sizeof(status));
    write(client_fd, &actual_size, sizeof(actual_size));
    if (status == CL_SUCCESS && actual_size > 0) {
        write(client_fd, buffer, actual_size);
    }
}

void handle_clCreateImage(int client_fd) {
    cl_mem_flags flags;
    uint32_t has_format = 0;
    cl_image_format format;
    uint32_t has_desc = 0;
    cl_image_desc desc;

    read(client_fd, &flags, sizeof(flags));

    read(client_fd, &has_format, sizeof(has_format));
    if (has_format) {
        read(client_fd, &format, sizeof(format));
    }

    read(client_fd, &has_desc, sizeof(has_desc));
    if (has_desc) {
        read(client_fd, &desc, sizeof(desc));
    }

    uint32_t no_shm = 0;
    read(client_fd, &no_shm, sizeof(no_shm));

    ShmHeader hdr{}; 
    int shm_fd = -1;
    bool got_shm = false;
    if(no_shm)
        got_shm = recv_fd_with_hdr(client_fd, hdr, shm_fd);

    void* host_ptr = nullptr;
    if (got_shm && shm_fd >= 0) {
        host_ptr = mmap(nullptr, hdr.size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

        if (!(flags & CL_MEM_USE_HOST_PTR)) {
            // Nothing needed here
        }
    } else {
        flags &= ~(CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR);
    }

    if (has_desc && desc.buffer != nullptr) {
        if (_cl_context) {
            ResourceId possible_mem_rid = reinterpret_cast<ResourceId>(desc.buffer); // Check if it is asking for a buffer that is created in advance
            if (mem_rid_to_cl_map[possible_mem_rid]) {
                desc.buffer = mem_rid_to_cl_map[possible_mem_rid];
            } else {
                desc.buffer = nullptr; // Otherwise NULL
            }
        }
    }

    cl_int status = CL_INVALID_CONTEXT;
    cl_mem _cl_image = nullptr;
    ResourceId image_rid = 0;

    auto clCreateImageFn = (cl_mem (*)(cl_context, cl_mem_flags, const cl_image_format*, const cl_image_desc*, void*, cl_int*)) load_vendor_func("clCreateImage");

    const cl_image_format* format_ptr = has_format ? &format : nullptr;
    const cl_image_desc* desc_ptr = has_desc ? &desc : nullptr;

    _cl_image = clCreateImageFn(_cl_context, flags, format_ptr, desc_ptr, host_ptr, &status);

    if (host_ptr && !(flags & CL_MEM_USE_HOST_PTR)) {
        munmap(host_ptr, hdr.size);
    }
    if (shm_fd >= 0) close(shm_fd);

    if (status == CL_SUCCESS) {
        image_rid = create_resource_id(RESOURCE_TYPE_IMAGE, mem_rid_counter++);
        mem_rid_to_cl_map[image_rid] = _cl_image;
    } else {
        LOG_ERROR("[SCHEDULER] clCreateImage failed: " << status);
    }

    write(client_fd, &status, sizeof(status));
    write(client_fd, &image_rid, sizeof(image_rid));
}

void handle_clCreateSubBuffer(int client_fd) {
    ResourceId parent_buffer_rid;
    cl_mem_flags flags;
    cl_buffer_create_type create_type;
    cl_buffer_region region;

    read(client_fd, &parent_buffer_rid, sizeof(parent_buffer_rid));
    read(client_fd, &flags, sizeof(flags));
    read(client_fd, &create_type, sizeof(create_type));
    read(client_fd, &region, sizeof(region));

    cl_int status = CL_INVALID_MEM_OBJECT;
    cl_mem _cl_parent_buffer = mem_rid_to_cl_map[parent_buffer_rid];
    cl_mem _cl_sub = nullptr;
    ResourceId sub_buffer_rid = 0;

    if (_cl_parent_buffer) {
        auto clCreateSubBufferFn = (cl_mem (*)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void*, cl_int*)) load_vendor_func("clCreateSubBuffer");

        _cl_sub = clCreateSubBufferFn(_cl_parent_buffer, flags, create_type, &region, &status);

        if (status == CL_SUCCESS) {
            sub_buffer_rid = create_resource_id(RESOURCE_TYPE_SUBMEM_OBJECT, mem_rid_counter++);
            mem_rid_to_cl_map[sub_buffer_rid] = _cl_sub;
        }
    }

    write(client_fd, &status, sizeof(status));
    write(client_fd, &sub_buffer_rid, sizeof(sub_buffer_rid));
}

void handle_clReleaseMemObject(int client_fd) {
    ResourceId mem_rid;
    read(client_fd, &mem_rid, sizeof(mem_rid));

    cl_mem _cl_mem = mem_rid_to_cl_map[mem_rid];
    cl_int status = CL_INVALID_MEM_OBJECT;

    if (_cl_mem) {
        auto clReleaseMemObjectFn = (cl_int (*)(cl_mem)) load_vendor_func("clReleaseMemObject");
        auto clGetMemObjectInfoFn = (cl_int (*)(cl_mem, cl_mem_info, size_t, void*, size_t*)) load_vendor_func("clGetMemObjectInfo");

        cl_uint ref_count = 1;
        clGetMemObjectInfoFn(_cl_mem, CL_MEM_REFERENCE_COUNT, sizeof(ref_count), &ref_count, nullptr);
        status = clReleaseMemObjectFn(_cl_mem);
        
        // Erase the object from the map if it is not referenced anymore
        if (ref_count == 1) {
            auto it = mem_rid_to_cl_map.find(mem_rid);
            mem_rid_to_cl_map.erase(it);
        }
    }

    write(client_fd, &status, sizeof(status));
}

void handle_clReleaseKernel(int client_fd) {
    ResourceId kernel_rid;
    read(client_fd, &kernel_rid, sizeof(kernel_rid));

    cl_kernel _cl_kernel = kernel_rid_to_cl_map[kernel_rid]._cl_kernel;
    cl_int status = CL_INVALID_KERNEL;

    if (_cl_kernel) {
        auto clReleaseKernelFn = (cl_int (*)(cl_kernel)) load_vendor_func("clReleaseKernel");
        auto clGetKernelInfoFn = (cl_int (*)(cl_kernel, cl_kernel_info, size_t, void*, size_t*)) load_vendor_func("clGetKernelInfo");

        cl_uint ref_count = 1;
        clGetKernelInfoFn(_cl_kernel, CL_KERNEL_REFERENCE_COUNT, sizeof(ref_count), &ref_count, nullptr);

        status = clReleaseKernelFn(_cl_kernel);

        // Erase the object from the map if it is not referenced anymore
        if (ref_count == 1) {
            auto it = kernel_rid_to_cl_map.find(kernel_rid);
            kernel_rid_to_cl_map.erase(it);
        }
    }

    write(client_fd, &status, sizeof(status));
}

void handle_clReleaseProgram(int client_fd) {
    ResourceId program_rid;
    read(client_fd, &program_rid, sizeof(program_rid));

    cl_program _cl_program = program_rid_to_cl_map[program_rid];
    cl_int status = CL_INVALID_PROGRAM;

    if (_cl_program) {
        auto clReleaseProgramFn = (cl_int (*)(cl_program)) load_vendor_func("clReleaseProgram");
        auto clGetProgramInfoFn = (cl_int (*)(cl_program, cl_program_info, size_t, void*, size_t*)) load_vendor_func("clGetProgramInfo");

        cl_uint ref_count = 1;
        clGetProgramInfoFn(_cl_program, CL_PROGRAM_REFERENCE_COUNT, sizeof(ref_count), &ref_count, nullptr);
        status = clReleaseProgramFn(_cl_program);
        
        // Erase the object from the map if it is not referenced anymore
        if (ref_count == 1) {
            auto it = program_rid_to_cl_map.find(program_rid);
            program_rid_to_cl_map.erase(it);
        }
    }

    write(client_fd, &status, sizeof(status));
}

// Releasing the command queue never happens, unless the OpenCL kernel scheduler is deleted
void handle_clReleaseCommandQueue(int client_fd) {
    ResourceId cmd_queue_rid;
    read(client_fd, &cmd_queue_rid, sizeof(cmd_queue_rid));

    cl_command_queue _cl_queue = nullptr;
    cl_int status = CL_SUCCESS;

    write(client_fd, &status, sizeof(status));
}

// Same as handle_clReleaseCommandQueue
void handle_clReleaseContext(int client_fd) {

    cl_int status = CL_SUCCESS;

    write(client_fd, &status, sizeof(status));
}

void handle_clRetainProgram(int client_fd) {
    ResourceId program_rid;
    read(client_fd, &program_rid, sizeof(program_rid));

    cl_program _cl_program = program_rid_to_cl_map[program_rid];
    cl_int status = CL_INVALID_PROGRAM;

    if (_cl_program) {
        auto clRetainProgramFn = (cl_int (*)(cl_program)) load_vendor_func("clRetainProgram");

        status = clRetainProgramFn(_cl_program);
    }

    write(client_fd, &status, sizeof(status));
}

// handle_clFlush is never invoked since we do not allow a user process flushing the commands
void handle_clFlush(int client_fd) {
    ResourceId cmd_queue_rid;
    read(client_fd, &cmd_queue_rid, sizeof(cmd_queue_rid));

    cl_int status = CL_SUCCESS;

    write(client_fd, &status, sizeof(status));
}

// clFinish is a blocking call, so we create a worker thread to handle it
void* finish_worker_thread(void* arg_ptr);
void handle_clFinish(int client_fd) {
    auto* arg = new int(client_fd);
    pthread_t tid;
    pthread_create(&tid, nullptr, finish_worker_thread, arg);
    pthread_detach(tid);
}

/* === End of Request Handling Functions === */

/* === CL_CALLBACK Functions === */

// Callback function for updating kernel execution time
void CL_CALLBACK kernel_execution_time_update_callback(cl_event event, cl_int exec_status, void* user_data) {
    if (exec_status != CL_COMPLETE) return;

    auto clGetEventProfilingInfoFn = (cl_int (*)(cl_event, cl_profiling_info, size_t, void*, size_t*)) load_vendor_func("clGetEventProfilingInfo");
    auto clReleaseEventFn = (cl_int (*)(cl_event)) load_vendor_func("clReleaseEvent");
   
    cl_kernel kernel = reinterpret_cast<cl_kernel>(user_data);
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfoFn(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    clGetEventProfilingInfoFn(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    cl_ulong duration_ns = (end - start);
    cl_ulong& current_avg = kernel_to_execution_time_map[kernel];
    uint32_t& count = kernel_to_execution_count_map[kernel];
    current_avg = (current_avg * count + duration_ns) / (count + 1);
    
    clReleaseEventFn(event); // Release the event to avoid memory leaks
}

/* === End of CL_CALLBACK Functions === */

/* === Threads === */

// A thread for handling clFinish per user process
// Dynamically created when a user process calls clFinish
void* finish_worker_thread(void* arg_ptr) {
    int client_fd = *(int*)arg_ptr;
    delete (int*)arg_ptr;

    ResourceId cmd_queue_rid;
    read(client_fd, &cmd_queue_rid, sizeof(cmd_queue_rid));

    cl_int status = CL_SUCCESS; // if there is no remaining kernels of that process, it will return success
    
    if (remaining_kernel_request_per_proc[cmd_queue_rid] > 0) {
        std::unique_lock<std::mutex> lock(clFinish_mutexes[cmd_queue_rid]);

        clFinish_cvars[cmd_queue_rid].wait(lock, [&] {
            return remaining_kernel_request_per_proc[cmd_queue_rid] == 0;
        });
    }

    write(client_fd, &status, sizeof(status));
    close(client_fd);

    return nullptr;
}

// Kernel Scheduling Thread
uint64_t MAX_KERNEL_EXECUTION_TIME_NS=1000000; // Variable for the maximum kernel execution time, initial value is 1 ms
void* kernel_scheduling_thread(void*){
    KernelRequestInfo kernel_request_info;
    uint64_t buf;                          // for eventfd reads
    static int count = 0;
    cl_ulong total_queued_kernel_execution_time_ns = 0;
    cl_ulong kernel_time_ns = 0;

    auto clFinishFn=(cl_int(*)(cl_command_queue))load_vendor_func("clFinish");
    auto clFlushFn = (cl_int (*)(cl_command_queue)) load_vendor_func("clFlush");
    auto clWaitForEventsFn = (cl_int (*)(cl_uint, const cl_event*)) load_vendor_func("clWaitForEvents");
    auto clEnqeueuNDRangeKernelFn=(cl_int(*)(cl_command_queue,cl_kernel,cl_uint,const size_t*,const size_t*,const size_t*,cl_uint,const cl_event*,cl_event*))load_vendor_func("clEnqueueNDRangeKernel");
    auto clEnqueueReadBufferFn=(cl_int(*)(cl_command_queue,cl_mem,cl_bool,size_t,size_t,void*,cl_uint,const cl_event*,cl_event*))load_vendor_func("clEnqueueReadBuffer");
    auto clEnqueueWriteBufferFn = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*)) load_vendor_func("clEnqueueWriteBuffer");
    auto clGetEventProfilingInfoFn = (cl_int (*)(cl_event, cl_profiling_info, size_t, void*, size_t*)) load_vendor_func("clGetEventProfilingInfo");
    auto clReleaseEventFn = (cl_int (*)(cl_event)) load_vendor_func("clReleaseEvent");
    auto clSetEventCallbackFn = (cl_int (*)(cl_event, cl_int, void (*)(cl_event, cl_int, void*), void*)) load_vendor_func("clSetEventCallback");

    while (true) {
        if (read(wake_fd, &buf, sizeof(buf)) < 0) {
            perror("eventfd read");
            LOG_INFO("[SCHEDULER] eventfd read failed");
            continue;
        }
    
        while(true){
            uint32_t sel_prio = get_highest_priority();

            if (sel_prio == PRIORITY_LEVELS) break; // No more work → go back to blocking

            PriorityKernelQueue<KernelRequestInfo>& rq = priority_kernel_queue[sel_prio];
            if (!rq.peek(kernel_request_info)) {    // rare case: bit set but the priority kernel queue is empty
                clear_priority_bit(sel_prio);
                continue;
            }

            if (rq.empty())  // if queue drained → clear bit
                clear_priority_bit(sel_prio);

            // Enqueue kernel
            if(kernel_request_info.type==TYPE_NDRANGE_KERNEL){                           
                // Profiling kernel execution time
                if(kernel_to_execution_time_map[kernel_request_info._cl_kernel] == 0) {
                    cl_event kernel_event;
                    clEnqeueuNDRangeKernelFn(_cl_command_queue,kernel_request_info._cl_kernel,kernel_request_info.work_dim,
                        kernel_request_info.global_offset.data(),
                        kernel_request_info.global_size.data(),
                        kernel_request_info.local_size.data(),
                        0,nullptr,&kernel_event);
                    cl_int status = clFlushFn(_cl_command_queue);
                    clWaitForEventsFn(1, &kernel_event);
                    cl_ulong start_time = 0, end_time = 0;
                    cl_ulong time_queued = 0, time_submit = 0;
                    clGetEventProfilingInfoFn(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
                    clGetEventProfilingInfoFn(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
                    clGetEventProfilingInfoFn(kernel_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &time_queued, nullptr);
                    clGetEventProfilingInfoFn(kernel_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &time_submit, nullptr);
                    cl_ulong kernel_time_ns = (end_time - start_time);
                    cl_ulong queuing_delay_ns = (time_submit - time_queued);
                    cl_ulong submit_delay_ns = (start_time - time_submit);
                    kernel_to_execution_time_map[kernel_request_info._cl_kernel] = kernel_time_ns;
                    clReleaseEventFn(kernel_event);

                    // Update MAX_KERNEL_EXECUTION_TIME_NS if needed
                    if(kernel_time_ns > MAX_KERNEL_EXECUTION_TIME_NS) {
                        MAX_KERNEL_EXECUTION_TIME_NS = (kernel_time_ns / 1000000) * 1000000; // round to ms
                        LOG_INFO("[SCHEDULER] Updated MAX_KERNEL_EXECUTION_TIME_NS to " << MAX_KERNEL_EXECUTION_TIME_NS / 1000000 << " ms");
                    }

                    LOG_INFO("[SCHEDULER] Kernel " << kernel_request_info._cl_kernel << " execution time: " << kernel_time_ns << " ns"
                                << " (queuing delay: " << queuing_delay_ns << " ns, dispatch delay: " << submit_delay_ns << " ns)");
                } else {
                    kernel_time_ns = kernel_to_execution_time_map[kernel_request_info._cl_kernel];

                    // Before enqueuing a kernel, we check if any higher priority kernel has been enqueued
                    uint32_t sel_prio_next = get_highest_priority();
                    if(sel_prio_next > sel_prio)
                        continue;
                    
                    // Case 1: If the total queued kernel execution time exceeds the maximum allowed time
                    if(total_queued_kernel_execution_time_ns + kernel_time_ns > MAX_KERNEL_EXECUTION_TIME_NS) {
                        // Case 1-1: If there is no kernel being executed nor waiting to be executed, 
                        // which means that this kernel has a longer execution time than the MAX_KERNEL_EXECUTION_TIME_NS
                        if(total_queued_kernel_execution_time_ns == 0){
                            // Case 1-1-1: Update kernel execution time
                            if(kernel_to_execution_count_map[kernel_request_info._cl_kernel]%50 == 0) {
                                cl_event kernel_update_event;
                                clEnqeueuNDRangeKernelFn(_cl_command_queue,kernel_request_info._cl_kernel,kernel_request_info.work_dim,
                                    kernel_request_info.global_offset.data(),
                                    kernel_request_info.global_size.data(),
                                    kernel_request_info.local_size.data(),
                                    0,nullptr, &kernel_update_event); //
                                clSetEventCallbackFn(kernel_update_event, CL_COMPLETE, kernel_execution_time_update_callback, kernel_request_info._cl_kernel);
                            } 
                            // Case 1-1-2: Enqueue the kernel and wait
                            else {
                                clEnqeueuNDRangeKernelFn(_cl_command_queue,kernel_request_info._cl_kernel,kernel_request_info.work_dim,
                                    kernel_request_info.global_offset.data(),
                                    kernel_request_info.global_size.data(),
                                    kernel_request_info.local_size.data(),
                                    0,nullptr,nullptr);
                            }
                            // Executing the largest kernel in the system
                            cl_int status = clFlushFn(_cl_command_queue);
                            std::this_thread::sleep_for(std::chrono::microseconds(kernel_time_ns/1000));
                        } 
                        // Case 1-2: There are kernels being exectued or waiting to be executed
                        else {
                            cl_int status = clFlushFn(_cl_command_queue);
                            std::this_thread::sleep_for(std::chrono::microseconds(total_queued_kernel_execution_time_ns/1000));
                            total_queued_kernel_execution_time_ns = 0; // reset after waiting
                            continue; // continue, since we do not enqueue the current one
                        }
                    } 
                    // Case 2: The total kernel execution time does not exceed the bound
                    // Enqueue the kernel and dispatch it immediately
                    else {
                        // Case 2-1: Update kernel execution time
                        if(kernel_to_execution_count_map[kernel_request_info._cl_kernel]%50 == 0) { // For updating kernel execution time
                            cl_event kernel_update_event;
                            clEnqeueuNDRangeKernelFn(_cl_command_queue,kernel_request_info._cl_kernel,kernel_request_info.work_dim,
                                kernel_request_info.global_offset.data(),
                                kernel_request_info.global_size.data(),
                                kernel_request_info.local_size.data(),
                                0,nullptr, &kernel_update_event);
                            clSetEventCallbackFn(kernel_update_event, CL_COMPLETE, kernel_execution_time_update_callback, kernel_request_info._cl_kernel);
                        } 
                        // Case 2-2: Enqueue the kernel
                        else {
                            clEnqeueuNDRangeKernelFn(_cl_command_queue,kernel_request_info._cl_kernel,kernel_request_info.work_dim,
                                kernel_request_info.global_offset.data(),
                                kernel_request_info.global_size.data(),
                                kernel_request_info.local_size.data(),
                                0,nullptr,nullptr);
                        }
                        total_queued_kernel_execution_time_ns += kernel_time_ns;
                        cl_int status = clFlushFn(_cl_command_queue);
                    }
                } 

                // After enqueuing the kernel, we update the kernel execution count
                kernel_to_execution_count_map[kernel_request_info._cl_kernel]++;
                // Now pop the kernel from the priority kernel queue
                rq.commit_pop();
                if(rq.empty()) {
                    clear_priority_bit(sel_prio);
                }
                // Wakeup clFinish threads if any exists
                remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid]--;
                if (remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid] == 0) {
                    std::lock_guard<std::mutex> lock(clFinish_mutexes[kernel_request_info.cmd_queue_rid]);
                    clFinish_cvars[kernel_request_info.cmd_queue_rid].notify_all();  // wake up all clFinish threads
                }
            }
            /* 
            * About read/write buffer commands: 
            * We also treat read/write buffer as a kernel, since it is a blocking call
            * With the DNN models we've used, the read/write duration does not exceed 2 ms even under GPU contention
            * So we fix the read/write time as 2 ms
            * Also, in unified memory architecture, which majority of AIoT systems have,
            * read/write buffer command can be totally eliminated by using buffer sharing
            */
            else if(kernel_request_info.type==TYPE_READ_BUFFER){
                if(total_queued_kernel_execution_time_ns + 2000000 > MAX_KERNEL_EXECUTION_TIME_NS && total_queued_kernel_execution_time_ns != 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(total_queued_kernel_execution_time_ns/1000));
                    total_queued_kernel_execution_time_ns = 0; // reset after waiting
                    continue;
                }
                
                void* map = mmap(nullptr, kernel_request_info.size, PROT_READ | PROT_WRITE, MAP_SHARED, kernel_request_info.shm_fd, 0);
                
                cl_int status = clEnqueueReadBufferFn(_cl_command_queue,kernel_request_info.buffer,kernel_request_info.blocking,kernel_request_info.offset,kernel_request_info.size,map,0,nullptr,nullptr);
                
                if(kernel_request_info.blocking == CL_NON_BLOCKING){
                    clFinishFn(_cl_command_queue);
                }
                total_queued_kernel_execution_time_ns = 0; // reset after waiting

                if(kernel_request_info.client_fd>=0){
                    write(kernel_request_info.client_fd,&status,sizeof(status));    
                    munmap(map, kernel_request_info.shm_size);
                    close(kernel_request_info.shm_fd);
                    close(kernel_request_info.client_fd);
                }
                
                rq.commit_pop();
                if(rq.empty()) {
                    clear_priority_bit(sel_prio);
                }

                remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid]--;
                if (remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid] == 0) {
                    std::lock_guard<std::mutex> lock(clFinish_mutexes[kernel_request_info.cmd_queue_rid]);
                    clFinish_cvars[kernel_request_info.cmd_queue_rid].notify_all();  // wake up all clFinish threads
                }
            }
            else if(kernel_request_info.type==TYPE_WRITE_BUFFER){
                if(total_queued_kernel_execution_time_ns + 2000000 > MAX_KERNEL_EXECUTION_TIME_NS && total_queued_kernel_execution_time_ns != 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(total_queued_kernel_execution_time_ns/1000));
                    total_queued_kernel_execution_time_ns = 0; // reset after waiting
                    continue;
                }

                void* map = mmap(nullptr, kernel_request_info.size, PROT_READ | PROT_WRITE, MAP_SHARED, kernel_request_info.shm_fd, 0);
                
                if (map == MAP_FAILED) {
                    LOG_ERROR("[SCHEDULER] mmap failed for WriteBuffer");
                    cl_int status = CL_OUT_OF_RESOURCES;
                    if(kernel_request_info.client_fd >= 0) write(kernel_request_info.client_fd, &status, sizeof(status));
                    close(kernel_request_info.shm_fd);
                    if(kernel_request_info.client_fd >= 0) close(kernel_request_info.client_fd);
                    continue;
                }

                cl_int status = clEnqueueWriteBufferFn(_cl_command_queue, kernel_request_info.buffer, kernel_request_info.blocking, kernel_request_info.offset, kernel_request_info.size, map, 0, nullptr, nullptr);

                if (kernel_request_info.blocking == CL_NON_BLOCKING) {
                    clFinishFn(_cl_command_queue);
                }
                total_queued_kernel_execution_time_ns = 0; // reset after waiting

                if(kernel_request_info.client_fd >= 0) {
                    munmap(map, kernel_request_info.shm_size);
                    close(kernel_request_info.shm_fd);
                }

                rq.commit_pop();
                if(rq.empty()) {
                    clear_priority_bit(sel_prio);
                }

                remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid]--;
                if (remaining_kernel_request_per_proc[kernel_request_info.cmd_queue_rid] == 0) {
                    std::lock_guard<std::mutex> lock(clFinish_mutexes[kernel_request_info.cmd_queue_rid]);
                    clFinish_cvars[kernel_request_info.cmd_queue_rid].notify_all();
                }
            }
        }
        // All queues empty → go back to blocking
        LOG_INFO("[SCHEDULER] All queues drained. Going back to sleep.");
    }
    return nullptr;
}

// Request reception thread is the main thread, which is defined below
/* === End of Threads === */


/* === Request Reception Thread === */

// The request reception thread handles incoming request from clients in a case-by-case manner
void receive_request(uint32_t req_type, int client_fd) {
    switch (req_type) {
        case REQ_GET_PLATFORM_IDS:
            handle_clGetPlatformIDs(client_fd);
            break;
        case REQ_GET_PLATFORM_INFO:
            handle_clGetPlatformInfo(client_fd);
            break;
        case REQ_GET_DEVICE_IDS:
            handle_clGetDeviceIDs(client_fd);
            break;
        case REQ_GET_DEVICE_INFO:
            handle_clGetDeviceInfo(client_fd);
            break;
        case REQ_CREATE_CONTEXT:
            handle_clCreateContext(client_fd);
            break;
        case REQ_CREATE_PROGRAM_WITH_SOURCE:
            handle_clCreateProgramWithSource(client_fd);
            break;
        case REQ_BUILD_PROGRAM:
            handle_clBuildProgram(client_fd);
            break;
        case REQ_CREATE_KERNEL:
            handle_clCreateKernel(client_fd);
            break;
        case REQ_CREATE_BUFFER:
            handle_clCreateBuffer(client_fd);
            break;
        case REQ_SET_KERNEL_ARG:
            handle_clSetKernelArg(client_fd);
            break;
        case REQ_CREATE_COMMAND_QUEUE_WITH_PROPERTIES:
            handle_clCreateCommandQueueWithProperties(client_fd);
            break;
        case REQ_ENQUEUE_WRITE_BUFFER:
            handle_clEnqueueWriteBuffer(client_fd);
            break;
        case REQ_ENQUEUE_NDRANGE_KERNEL:
            handle_clEnqueueNDRangeKernel(client_fd);
            break;
        case REQ_ENQUEUE_READ_BUFFER:
            handle_clEnqueueReadBuffer(client_fd);
            break;
        case REQ_GET_SUPPORTED_IMAGE_FORMATS:
            handle_clGetSupportedImageFormats(client_fd);
            break;
        case REQ_GET_KERNEL_WORK_GROUP_INFO:
            handle_clGetKernelWorkGroupInfo(client_fd);
            break;
        case REQ_CREATE_IMAGE:
            handle_clCreateImage(client_fd);
            break;
        case REQ_CREATE_SUBBUFFER:
            handle_clCreateSubBuffer(client_fd);
            break;
        case REQ_RELEASE_MEM_OBJECT:
            handle_clReleaseMemObject(client_fd);
            break;
        case REQ_RELEASE_KERNEL:
            handle_clReleaseKernel(client_fd);
            break;
        case REQ_RELEASE_PROGRAM:
            handle_clReleaseProgram(client_fd);
            break;
        case REQ_RELEASE_COMMAND_QUEUE:
            handle_clReleaseCommandQueue(client_fd);
            break;
        case REQ_RELEASE_CONTEXT:
            handle_clReleaseContext(client_fd);
            break;
        case REQ_RETAIN_PROGRAM:
            handle_clRetainProgram(client_fd);
            break;
        case REQ_FLUSH:
            handle_clFlush(client_fd);
            break;
        case REQ_FINISH:
            handle_clFinish(client_fd);
            break;
        default:
            LOG_ERROR("[SCHEDULER] Unknown request type: " << req_type);
            break;
    }
}

int main() {
    // init scheduler
    init_scheduler();
    
    // Ignore SIGPIPE to prevent crashes on broken pipes
    signal(SIGPIPE, SIG_IGN);

    // Initializing kernel scheduling thread
    wake_fd = eventfd(0, EFD_SEMAPHORE);
    if (wake_fd < 0) { perror("eventfd"); return 1; }
    pthread_t kernel_scheduler_tid;
    LOG_INFO("[SCHEDULER] Starting kernel scheduler thread");
    pthread_create(&kernel_scheduler_tid, nullptr, kernel_scheduling_thread, nullptr);

    // Open up socket to receive requests
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    unlink(SOCKET_PATH);
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    listen(server_fd, 5);
    LOG_INFO("[SCHEDULER] Listening on " << SOCKET_PATH);

    // This while loop acts as the request reception thread
    while (true) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;

        uint32_t req_type = 0;
        ssize_t r = read(client_fd, &req_type, sizeof(req_type));
        if (r == sizeof(req_type)) {
            receive_request(req_type, client_fd);
        } else {
            LOG_ERROR("[SCHEDULER] Failed to read request type.");
        }

        // We do not close the client_fd for REQ_ENQUEUE_READ_BUFFER and REQ_FINISH
        // because they are handled in the kernel_scheduling_thread
        if(req_type != REQ_ENQUEUE_READ_BUFFER && req_type != REQ_FINISH)
            close(client_fd);
    }

    return 0;
}
/* === End of Request Reception Thread === */

/* === END OF FILE === */
