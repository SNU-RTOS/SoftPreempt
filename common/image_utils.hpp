/********************************************************
 * Author: Namcheol Lee
 * Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * Contact: nclee@redwood.snu.ac.kr
 * Date: 2025-07-02
 * Description: Utility functions for image size calculation for clCreateImage()
 ********************************************************/


size_t get_image_element_size(const cl_image_format* format) {
    size_t channel_count;
    size_t channel_size;

    switch (format->image_channel_order) {
        case CL_R: channel_count = 1; break;
        case CL_RG: channel_count = 2; break;
        case CL_RGB: channel_count = 3; break;
        case CL_RGBA: channel_count = 4; break;
        case CL_BGRA: channel_count = 4; break;
        case CL_ARGB: channel_count = 4; break;
        default: channel_count = 1; break; // fallback
    }

    switch (format->image_channel_data_type) {
        case CL_UNORM_INT8:
        case CL_SNORM_INT8:
        case CL_UNSIGNED_INT8:
        case CL_SIGNED_INT8:
            channel_size = 1; break;
        case CL_UNORM_INT16:
        case CL_SNORM_INT16:
        case CL_UNSIGNED_INT16:
        case CL_SIGNED_INT16:
        case CL_HALF_FLOAT:
            channel_size = 2; break;
        case CL_UNSIGNED_INT32:
        case CL_SIGNED_INT32:
        case CL_FLOAT:
            channel_size = 4; break;
        default:
            channel_size = 1; break; // fallback
    }

    return channel_count * channel_size;
}

size_t calculate_image_size(const cl_image_format* format, const cl_image_desc* desc) {
    size_t element_size = get_image_element_size(format);

    size_t row_pitch = desc->image_row_pitch ? desc->image_row_pitch
                                              : desc->image_width * element_size;
    size_t slice_pitch = desc->image_slice_pitch ? desc->image_slice_pitch
                                                  : row_pitch * desc->image_height;

    switch (desc->image_type) {
        case CL_MEM_OBJECT_IMAGE1D:
            return row_pitch;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        case CL_MEM_OBJECT_IMAGE2D:
            return row_pitch * desc->image_height;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        case CL_MEM_OBJECT_IMAGE3D:
            return slice_pitch * desc->image_depth;
        default:
            return 0;
    }
}