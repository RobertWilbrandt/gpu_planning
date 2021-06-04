#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER gpu_planning

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "tracepoints.hpp"

#if !defined(GPU_PLANNING_TRACEPOINTS_TRACEPOINTS_HPP) || \
    defined(TRACEPOINT_HEADER_MULTI_READ)
#define GPU_PLANNING_TRACEPOINTS_TRACEPOINTS_HPP

#include <lttng/tracepoint.h>

TRACEPOINT_EVENT(
    TRACEPOINT_PROVIDER, map_creation,
    TP_ARGS(float, width_arg, float, height_arg, size_t, resolution_arg),
    TP_FIELDS(ctf_float(float, width, width_arg)
                  ctf_float(float, height, height_arg)
                      ctf_integer(size_t, resolution, resolution_arg)))

TRACEPOINT_EVENT(TRACEPOINT_PROVIDER, work_buffer_block_dispatch,
                 TP_ARGS(size_t, size_arg, size_t, block_size_arg, size_t,
                         offset_arg),
                 TP_FIELDS(ctf_integer(size_t, size, size_arg)
                               ctf_integer(size_t, block_size, block_size_arg)
                                   ctf_integer(size_t, offset, offset_arg)))

TRACEPOINT_EVENT(TRACEPOINT_PROVIDER, collision_check_configuration,
                 TP_ARGS(size_t, num_arg),
                 TP_FIELDS(ctf_integer(size_t, num, num_arg)))

TRACEPOINT_EVENT(TRACEPOINT_PROVIDER, collision_check_segment,
                 TP_ARGS(size_t, num_arg),
                 TP_FIELDS(ctf_integer(size_t, num, num_arg)))

TRACEPOINT_EVENT(TRACEPOINT_PROVIDER, my_first_tracepoint,
                 TP_ARGS(int, my_integer_arg, char*, my_string_arg),
                 TP_FIELDS(ctf_string(my_string_field, my_string_arg)
                               ctf_integer(int, my_integer_field,
                                           my_integer_arg)))

#endif  // if !defined(GPU_PLANNING_TRACEPOINTS_TRACEPOINTS_HPP) ||
        // defined(TRACEPOINT_HEADER_MULTI_READ)

#include <lttng/tracepoint-event.h>

