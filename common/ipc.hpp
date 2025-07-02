/********************************************************
 * Author: Namcheol Lee
 * Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * Contact: nclee@redwood.snu.ac.kr
 * Date: 2025-07-02
 * Description: Data structures for IPC
 ********************************************************/
 
#pragma once
#include <stdint.h>
#include <stddef.h>

#define SOCKET_PATH "/tmp/opencl_scheduler.sock"
typedef uint64_t ResourceId;
#define PLATFORM_RESOURCE_ID ((ResourceId)0xdead1000)
#define DEVICE_RESOURCE_ID ((ResourceId)0xdead2000)
#define CONTEXT_RESOURCE_ID ((ResourceId)0xdead3000)

// request types
enum RequestType : uint32_t {
    REQ_GET_PLATFORM_IDS = 1,
    REQ_GET_PLATFORM_INFO = 2,
    REQ_GET_DEVICE_IDS = 3,
    REQ_GET_DEVICE_INFO = 4,
    REQ_CREATE_CONTEXT = 5,
    REQ_CREATE_PROGRAM_WITH_SOURCE = 6,
    REQ_BUILD_PROGRAM = 7,
    REQ_CREATE_KERNEL = 8,
    REQ_CREATE_BUFFER = 9,
    REQ_SET_KERNEL_ARG = 10,
    REQ_CREATE_COMMAND_QUEUE_WITH_PROPERTIES = 11,
    REQ_ENQUEUE_WRITE_BUFFER = 12,
    REQ_ENQUEUE_NDRANGE_KERNEL = 13,
    REQ_ENQUEUE_READ_BUFFER = 14,
    REQ_GET_SUPPORTED_IMAGE_FORMATS = 15,
    REQ_GET_KERNEL_WORK_GROUP_INFO = 16,
    REQ_CREATE_IMAGE = 17,
    REQ_CREATE_SUBBUFFER = 18,
    REQ_RELEASE_MEM_OBJECT = 19,
    REQ_RELEASE_KERNEL = 20,
    REQ_RELEASE_PROGRAM = 21,
    REQ_RELEASE_COMMAND_QUEUE = 22,
    REQ_RELEASE_CONTEXT = 23,
    REQ_RETAIN_PROGRAM = 24,
    REQ_FLUSH = 25,
    REQ_FINISH = 26,
};

// Shared memory types
enum ShmType : uint32_t { 
    SHM_TYPE_BUFFER = 1, 
    SHM_TYPE_IMAGE = 2, 
    SHM_TYPE_KERNEL = 3,
    SHM_TYPE_KERNEL_BATCH = 4,
 };

 // Shared memory header
struct ShmHeader {
    uint32_t type;            // ShmKind
    uint32_t flags;           // CL flags (only CL_MEM_* bits)
    size_t   size;            // bytes in mapping
    uint32_t extra;           // reserved (0 for buffer, depth for image, …)
};

// Kernel argument structure
struct KernelArg {
    cl_uint index;
    size_t size;
    uint8_t value[64]; // 64 bytes is enough for most types
};

// Kernel enqueue information sent from the OpenCL shim
struct KernelEnqueueInfo {
    pid_t pid;
    ResourceId cmd_queue_rid;
    ResourceId kernel_rid;
    cl_uint work_dim;
    size_t global_offset[3];
    size_t global_work_size[3];
    size_t local_work_size[3];
    cl_uint num_args;
};