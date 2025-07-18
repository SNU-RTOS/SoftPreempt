/********************************************************
 * Author: Namcheol Lee
 * Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * Contact: nclee@redwood.snu.ac.kr
 * Date: 2025-07-02
 * Description: OpenCL shim for profiling purpose
 ********************************************************/
 
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
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

#include "logging.hpp"

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

void* vendor_opencl_lib = nullptr;

void* load_vendor_func(const char* name) {
    // just for safety, check if the OpenCL library is loaded
    if (!vendor_opencl_lib) {
        vendor_opencl_lib = dlopen("/usr/lib/libOpenCL_vendor.so", RTLD_NOW | RTLD_LOCAL); // /opt/qti/usr/lib/libOpenCL.so.adreno
        if (!vendor_opencl_lib) {
            std::cerr << "[SHIM] dlopen failed: " << std::endl;
            exit(1);
        }
    }
    void* sym = dlsym(vendor_opencl_lib, name);
    if (!sym) {
        std::cerr << "[SHIM] dlsym failed" << std::endl;
        exit(1);
    }
    return sym;
}

extern "C" {   // <-- C linkage to avoid function name mangling of g++ compiler
// GCC extension
// The function is called when the library is loaded
__attribute__((constructor))
void on_shim_load() {
    std::cout << "[SHIM] libOpenCL_shim_internal.so loaded!\n" << std::endl;

    // Load the vendor-provided OpenCL library
    if (!vendor_opencl_lib) {
        vendor_opencl_lib = dlopen("/usr/lib/libOpenCL_vendor.so", RTLD_NOW | RTLD_LOCAL);
        if (!vendor_opencl_lib) {
            std::cerr << "[SCHEDULER] dlopen failed: " << std::endl;
            exit(1);
        }
    }
}

// Function pointer
void* clGetExtensionFunctionAddress(const char* func_name) {
    if (!func_name) {
        std::cerr << "[SHIM] clGetExtensionFunctionAddress called with NULL func_name" << std::endl;
        return nullptr;
    }

    std::cout << "[SHIM] clGetExtensionFunctionAddress called for: " << func_name << std::endl;

    for (const FunctionEntry* entry = scheduler_function_table; entry->name != nullptr; ++entry) {
        if (strcmp(entry->name, func_name) == 0) {
            std::cout << "[SHIM] Found function: " << func_name << " → " << entry->pointer << std::endl;
            return entry->pointer;
        }
    }

    std::cerr << "[SHIM] Function not found: " << func_name << std::endl;
    return nullptr;
}

void* clGetExtensionFunctionAddressForPlatform(cl_platform_id platform, const char* func_name) {
    std::cout << "[SHIM] clGetExtensionFunctionAddressForPlatform called for: " << func_name << std::endl;
    return clGetExtensionFunctionAddress(func_name);
}

// ===== Start of Platform Functions =====
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    cl_int status;
    auto clGetPlatformIDsFn = (cl_int (*)(cl_uint, cl_platform_id*, cl_uint*)) load_vendor_func("clGetPlatformIDs");
    status = clGetPlatformIDsFn(num_entries, platforms, num_platforms);

    return status;
}

cl_int clGetPlatformInfo(cl_platform_id platform,
                         cl_platform_info param_name,
                         size_t param_value_size,
                         void* param_value,
                         size_t* param_value_size_ret) {
    cl_int status;
    auto clGetPlatformInfoFn = (cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*)) load_vendor_func("clGetPlatformInfo");

    status = clGetPlatformInfoFn(platform, param_name, param_value_size, param_value, param_value_size_ret);

    return status;
}
// ===== End of Platform Functions =====

// ===== Start of Device Functions =====
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
    cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) {
    cl_int status;

    auto clGetDeviceIDsFn = (cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*)) load_vendor_func("clGetDeviceIDs");

    status = clGetDeviceIDsFn(platform, device_type, num_entries, devices, num_devices);

    return status;
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    cl_int status;

    auto clGetDeviceInfoFn = (cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*)) load_vendor_func("clGetDeviceInfo");
    status = clGetDeviceInfoFn(device, param_name, param_value_size, param_value, param_value_size_ret);

    return status;
}

// cl_context is a pointer of _cl_context struct
cl_context clCreateContext(const cl_context_properties *properties,
    cl_uint num_devices,
    const cl_device_id *devices,
    void (*pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) {
    
    auto clCreateContextFn = (cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*)) load_vendor_func("clCreateContext");
    
    return clCreateContextFn(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

cl_program clCreateProgramWithSource(cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret) {

    auto clCreateProgramWithSourceFn = (cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*)) load_vendor_func("clCreateProgramWithSource");

    return clCreateProgramWithSourceFn(context, count, strings, lengths, errcode_ret);
}

cl_int clBuildProgram(cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (*pfn_notify)(cl_program, void*),
    void* user_data) {

    cl_int status;
    auto clBuildProgramFn = (cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*)) load_vendor_func("clBuildProgram");
    
    status = clBuildProgramFn(program, num_devices, device_list, options, pfn_notify, user_data);

    return status;
}

cl_kernel clCreateKernel(cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret) {
    
    auto clCreateKernelFn = (cl_kernel (*)(cl_program, const char*, cl_int*)) load_vendor_func("clCreateKernel");

    return clCreateKernelFn(program, kernel_name, errcode_ret);
}

cl_mem clCreateBuffer(cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret) {

    auto clCreateBufferFn = (cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*)) load_vendor_func("clCreateBuffer");

    return clCreateBufferFn(context, flags, size, host_ptr, errcode_ret);
}

cl_int clSetKernelArg(cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value) {
    
    cl_int status;
    auto clSetKernelArgFn = (cl_int (*)(cl_kernel, cl_uint, size_t, const void*))load_vendor_func("clSetKernelArg");
    status = clSetKernelArgFn(kernel, arg_index, arg_size, arg_value);

    return status;
}

// Note: We manually create a command queue with properties
cl_command_queue clCreateCommandQueueWithProperties(cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret) {

    auto clCreateCommandQueueWithPropertiesFn = (cl_command_queue (*)(cl_context, cl_device_id, const cl_queue_properties*, cl_int*)) load_vendor_func("clCreateCommandQueueWithProperties");

    // Check if properties already enable profiling
    bool profiling_enabled = false;
    size_t count = 0;

    if (properties) {
        for (const cl_queue_properties* p = properties; p[0] != 0; p += 2) {
            if (p[0] == CL_QUEUE_PROPERTIES &&
                (p[1] & CL_QUEUE_PROFILING_ENABLE)) {
                profiling_enabled = true;
                break;
            }
            count += 2;
        }
    }

    if (profiling_enabled) {
        // Use original properties
        return clCreateCommandQueueWithPropertiesFn(context, device, properties, errcode_ret);
    } else {
        // Copy and add profiling flag
        cl_queue_properties* new_props = new cl_queue_properties[count + 4]; // 2 more pairs + null
        bool found = false;

        for (size_t i = 0; i < count; i += 2) {
            new_props[i] = properties[i];
            new_props[i + 1] = properties[i + 1];

            if (new_props[i] == CL_QUEUE_PROPERTIES) {
                new_props[i + 1] |= CL_QUEUE_PROFILING_ENABLE;
                found = true;
            }
        }

        if (!found) {
            new_props[count++] = CL_QUEUE_PROPERTIES;
            new_props[count++] = CL_QUEUE_PROFILING_ENABLE;
        }

        new_props[count++] = 0; // null-terminate

        cl_command_queue queue = clCreateCommandQueueWithPropertiesFn(context, device, new_props, errcode_ret);
        delete[] new_props;
        return queue;
    }
}

// Note: This function is deprecated since OpenCL 2.0
// It is redirected to clCreateCommandQueueWithProperties
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

void CL_CALLBACK time_profiling_callback(cl_event event, cl_int exec_status, void* user_data) {
    if (exec_status != CL_COMPLETE) return;

    auto clGetEventProfilingInfoFn = (cl_int (*)(cl_event, cl_profiling_info, size_t, void*, size_t*)) load_vendor_func("clGetEventProfilingInfo");
    auto clReleaseEventFn = (cl_int (*)(cl_event)) load_vendor_func("clReleaseEvent");

    cl_ulong queued = 0, submitted = 0, started = 0, ended = 0;
    clGetEventProfilingInfoFn(event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, nullptr);
    clGetEventProfilingInfoFn(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(submitted), &submitted, nullptr);
    clGetEventProfilingInfoFn(event, CL_PROFILING_COMMAND_START,  sizeof(started), &started, nullptr);
    clGetEventProfilingInfoFn(event, CL_PROFILING_COMMAND_END,    sizeof(ended), &ended, nullptr);

    int kernel_index = *(static_cast<int*>(user_data));

    printf("[PROFILE] Kernel #%d:\n", kernel_index);
    printf("  Queued: %.3f \n", (queued) / 1e6);
    printf("  Submit: %.3f \n", (submitted) / 1e6);
    printf("  Start: %.3f \n", (started) / 1e6);
    printf("  Ended: %.3f \n", (ended) / 1e6);
    printf("  Duration:\n");
    printf("  Queued → Submit : %.3f ms\n", (submitted - queued) / 1e6);
    printf("  Submit → Start  : %.3f ms\n", (started  - submitted) / 1e6);
    printf("  Start  → End    : %.3f ms\n", (ended    - started) / 1e6);
    printf("  Total           : %.3f ms\n", (ended    - queued) / 1e6);

    delete static_cast<int*>(user_data);
    clReleaseEventFn(event);
}

cl_int clEnqueueWriteBuffer(cl_command_queue queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events,
    const cl_event* event_wait_list,
    cl_event* event) {
    
    cl_int status = CL_SUCCESS;
    auto clEnqueueWriteBufferFn = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*)) load_vendor_func("clEnqueueWriteBuffer");
    status = clEnqueueWriteBufferFn(queue, buffer, blocking_write, offset, size, ptr, num_events, event_wait_list, event);

    return status;
}

static int kernel_counter = 0;
cl_int clEnqueueNDRangeKernel(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events,
    const cl_event* event_wait_list,
    cl_event* event) {
    
    cl_int status;
    auto clEnqueueNDRangeKernelFn=(cl_int(*)(cl_command_queue,cl_kernel,cl_uint,const size_t*,const size_t*,const size_t*,cl_uint,const cl_event*,cl_event*))load_vendor_func("clEnqueueNDRangeKernel");
    auto clSetEventCallbackFn = (cl_int (*)(cl_event, cl_int, void (*)(cl_event, cl_int, void*), void*)) load_vendor_func("clSetEventCallback");
    
    if (event) {
        // User wants the event — just forward the call, no profiling
        status = clEnqueueNDRangeKernelFn(
            queue, kernel, work_dim,
            global_work_offset, global_work_size, local_work_size,
            num_events, event_wait_list, event
        );
    } else {
        // User doesn't want the event — create one for profiling
        cl_event profiling_event = nullptr;

        status = clEnqueueNDRangeKernelFn(
            queue, kernel, work_dim,
            global_work_offset, global_work_size, local_work_size,
            num_events, event_wait_list, &profiling_event
        );

        if (status == CL_SUCCESS && profiling_event) {
            int* index_ptr = new int(kernel_counter++);
            clSetEventCallbackFn(profiling_event, CL_COMPLETE, time_profiling_callback, index_ptr);
            // DO NOT release here — will be released in the callback
        }
    }
    
    
    return status;
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

    cl_int status;
    auto clEnqueueReadBufferFn=(cl_int(*)(cl_command_queue,cl_mem,cl_bool,size_t,size_t,void*,cl_uint,const cl_event*,cl_event*))load_vendor_func("clEnqueueReadBuffer");
    status = clEnqueueReadBufferFn(queue, buffer, blocking_read, offset, size, ptr, num_events, event_wait_list, event);
    kernel_counter = 0;

    return status;
}

cl_int clGetSupportedImageFormats(cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats) {

    cl_int status;
    auto clGetSupportedImageFormatsFn = (cl_int (*)(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format*, cl_uint*)) load_vendor_func("clGetSupportedImageFormats");
    status = clGetSupportedImageFormatsFn(context, flags, image_type, num_entries, image_formats, num_image_formats);

    return status;
}

cl_int clGetKernelWorkGroupInfo(cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) {

    cl_int status;
    auto clGetKernelWorkGroupInfoFn = (cl_int (*)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*)) load_vendor_func("clGetKernelWorkGroupInfo");
    status = clGetKernelWorkGroupInfoFn(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);

    return status;
}

cl_mem clCreateImage(cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret) {
    
    auto clCreateImageFn = (cl_mem (*)(cl_context, cl_mem_flags, const cl_image_format*, const cl_image_desc*, void*, cl_int*)) load_vendor_func("clCreateImage");

    return clCreateImageFn(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

cl_mem clCreateSubBuffer(cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type create_type,
    const void* create_info,
    cl_int* errcode_ret) {

    auto clCreateSubBufferFn = (cl_mem (*)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void*, cl_int*)) load_vendor_func("clCreateSubBuffer");

    return clCreateSubBufferFn(buffer, flags, create_type, create_info, errcode_ret);
}

cl_int clReleaseMemObject(cl_mem memobj) {
    cl_int status;
    auto clReleaseMemObjectFn = (cl_int (*)(cl_mem)) load_vendor_func("clReleaseMemObject");
    status = clReleaseMemObjectFn(memobj);

    return status;
}

cl_int clReleaseKernel(cl_kernel kernel) {
    cl_int status;
    auto clReleaseKernelFn = (cl_int (*)(cl_kernel)) load_vendor_func("clReleaseKernel");
    status = clReleaseKernelFn(kernel);

    return status;
}

cl_int clReleaseProgram(cl_program program) {
    cl_int status;
    auto clReleaseProgramFn = (cl_int (*)(cl_program)) load_vendor_func("clReleaseProgram");
    status = clReleaseProgramFn(program);

    return status;
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    cl_int status;
    auto clReleaseCommandQueueFn = (cl_int (*)(cl_command_queue)) load_vendor_func("clReleaseCommandQueue");
    status = clReleaseCommandQueueFn(command_queue);

    return status;
}

cl_int clReleaseContext(cl_context context) {
    cl_int status;
    auto clReleaseContextFn = (cl_int (*)(cl_context)) load_vendor_func("clReleaseContext");
    status = clReleaseContextFn(context);
    
    return status;
}

cl_int clRetainProgram(cl_program program) {
    cl_int status;
    auto clRetainProgramFn = (cl_int (*)(cl_program)) load_vendor_func("clRetainProgram");
    status = clRetainProgramFn(program);

    return status;
}

cl_int clFlush(cl_command_queue command_queue) {
    cl_int status;
    auto clFlushFn = (cl_int(*)(cl_command_queue)) load_vendor_func("clFlush");
    status = clFlushFn(command_queue);

    return status;
}

cl_int clFinish(cl_command_queue command_queue) {
    cl_int status;
    auto clFinishFn = (cl_int(*)(cl_command_queue)) load_vendor_func("clFinish");
    status = clFinishFn(command_queue);

    return status;
}
// ===== End of Device Functions =====


} // end of extern "C" 