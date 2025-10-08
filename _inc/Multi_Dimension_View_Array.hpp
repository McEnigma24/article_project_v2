#include "__preprocessor__.h"
#include "parallel_common.h"
#include <vector>
#include <span>

// #define BOUND_CHECKS_Multi_Dimension_View_Array(...) __VA_ARGS__
#define BOUND_CHECKS_Multi_Dimension_View_Array(...)

struct coords
{
    u64 x, y, z;
};

template <typename T>
class Multi_Dimension_View_Array
{
    u64 WIDTH;
    u64 HEIGHT;
    u64 DEPTH;
    T* buffer_ptr;
    size_t buffer_size;

    GPU_LINE(__host__ __device__) bool bound_check(u64 x, u64 y, u64 z) const { return ((0 < x && x < WIDTH) && (0 < y && y < HEIGHT) && (0 < z && z < DEPTH)) && bound_check(to_d1(x, y, z)); }
    GPU_LINE(__host__ __device__) bool bound_check(u64 x, u64 y) const { return ((0 < x && x < WIDTH) && (0 < y && y < HEIGHT)) && bound_check(to_d1(x, y)); }
    GPU_LINE(__host__ __device__) bool bound_check(u64 x) const { return (0 < x && x < buffer_size); }

    GPU_LINE(__host__ __device__) u64 to_d1(u64 x, u64 y, u64 z) const { return x + y * (WIDTH) + z * (WIDTH * HEIGHT); }
    GPU_LINE(__host__ __device__) u64 to_d1(u64 x, u64 y) const { return x + y * (WIDTH); }

public:
    GPU_LINE(__host__ __device__)
    Multi_Dimension_View_Array() : WIDTH(0), HEIGHT(0), DEPTH(0), buffer_ptr(nullptr), buffer_size(0) {}
    ~Multi_Dimension_View_Array()
    {
        CPU_LINE(if(buffer_ptr) delete[] buffer_ptr);
    }

    GPU_LINE(__host__ __device__)
    void init(i64 width, i64 height, i64 depth, T* preallocated_buffer = nullptr)
    {
        CPU_LINE(if (width <= 0) FATAL_ERROR("WIDTH set with incorrect value"));
        CPU_LINE(if (height <= 0) FATAL_ERROR("HEIGHT set with incorrect value"));
        CPU_LINE(if (depth <= 0) FATAL_ERROR("DEPTH set with incorrect value"));

        WIDTH = width;
        HEIGHT = height;
        DEPTH = depth;
        buffer_size = WIDTH * HEIGHT * DEPTH;

        CPU_LINE(if(buffer_ptr) delete[] buffer_ptr);
        if(preallocated_buffer)
        {
            buffer_ptr = preallocated_buffer;
        }
        else { buffer_ptr = new T[buffer_size]; }
    }

    GPU_LINE(__host__ __device__) u64 get_width() const { return WIDTH; }
    GPU_LINE(__host__ __device__) u64 get_height() const { return HEIGHT; }
    GPU_LINE(__host__ __device__) u64 get_depth() const { return DEPTH; }
    GPU_LINE(__host__ __device__) u64 get_total_number() const { return buffer_size; }
    GPU_LINE(__host__ __device__) T* get_data() { return buffer_ptr; }

    GPU_LINE(__host__ __device__)
    T* get(u64 x, u64 y, u64 z)
    {
        CPU_LINE(BOUND_CHECKS_Multi_Dimension_View_Array(if (!bound_check(x, y, z)) FATAL_ERROR("accesing beyond vector bounds");))

            return &buffer_ptr[to_d1(x, y, z)];
    }

    GPU_LINE(__host__ __device__)
    T* get(u64 x, u64 y)
    {
        CPU_LINE(BOUND_CHECKS_Multi_Dimension_View_Array(if (!bound_check(x, y)) FATAL_ERROR("accesing beyond vector bounds");))

            return &buffer_ptr[to_d1(x, y)];
    }

    GPU_LINE(__host__ __device__)
    T* get(u64 x)
    {
        CPU_LINE(BOUND_CHECKS_Multi_Dimension_View_Array(if (!(x < buffer_size)) FATAL_ERROR("accesing beyond vector bounds");))

            return &buffer_ptr[x];
    }

    GPU_LINE(__host__ __device__)
    coords get_coords(u64 index_1d) const
    {
        CPU_LINE(BOUND_CHECKS_Multi_Dimension_View_Array(if (!(index_1d < buffer_size)) FATAL_ERROR("accesing beyond vector bounds");))

        coords c;
        c.z = index_1d / (WIDTH * HEIGHT);
        c.y = (index_1d - c.z * (WIDTH * HEIGHT)) / WIDTH;
        c.x = index_1d - c.z * (WIDTH * HEIGHT) - c.y * WIDTH;

        return c;
    }
};
