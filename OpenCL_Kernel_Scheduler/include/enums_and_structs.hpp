/********************************************************
 * Author: Namcheol Lee
 * Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * Contact: nclee@redwood.snu.ac.kr
 * Date: 2025-07-02
 * Description: Enums and Structs used in OpenCL Kernel Scheduler
 ********************************************************/

// Resource types that are managed internally
enum ResourceType : uint32_t {
    RESOURCE_TYPE_CONTEXT = 0x01,
    RESOURCE_TYPE_PROGRAM = 0x02,
    RESOURCE_TYPE_KERNEL = 0x03,
    RESOURCE_TYPE_COMMAND_QUEUE = 0x04,
    RESOURCE_TYPE_MEM_OBJECT = 0x05,
    RESOURCE_TYPE_SUBMEM_OBJECT = 0x06,
    RESOURCE_TYPE_IMAGE = 0x07,
    RESOURCE_TYPE_EVENT = 0x08, // not in use
    RESOURCE_TYPE_SAMPLER = 0x09, // not in use
    RESOURCE_TYPE_QUEUE = 0x11,
};

// Enum for Command management
enum CommandType : uint32_t {
    TYPE_NDRANGE_KERNEL = 1,
    TYPE_WRITE_BUFFER = 2,
    TYPE_READ_BUFFER = 3,
};

// Structure enqueued in the priority kernel queue
struct KernelRequestInfo {
    CommandType type;
    ResourceId cmd_queue_rid;

    // For NDRange kernel
    cl_kernel _cl_kernel = nullptr;
    cl_uint work_dim = 0;
    std::vector<size_t> global_offset;
    std::vector<size_t> global_size;
    std::vector<size_t> local_size;
    cl_uint num_events_in_wait_list = 0;
    const cl_event* event_wait_list = nullptr;
    cl_event* event = nullptr;

    // For Read / Write Buffer
    cl_mem  buffer   = nullptr;
    size_t  offset   = 0;
    size_t  size     = 0;
    cl_bool blocking = CL_TRUE;

    // For reply after read buffer
    int client_fd = -1;
    int shm_fd = -1;
    size_t shm_size = 0;
};

// Holds information about the kernel and its name
struct KernelInfo {
    cl_kernel _cl_kernel;
    std::string kernel_name;
    // args info, etc.
};