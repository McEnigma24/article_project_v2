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
    vector<T> buffer;

    GPU_LINE(__host__) bool bound_check(u64 x, u64 y, u64 z) const { return ((0 < x && x < WIDTH) && (0 < y && y < HEIGHT) && (0 < z && z < DEPTH)) && bound_check(to_d1(x, y, z)); }
    GPU_LINE(__host__) bool bound_check(u64 x, u64 y) const { return ((0 < x && x < WIDTH) && (0 < y && y < HEIGHT)) && bound_check(to_d1(x, y)); }
    GPU_LINE(__host__) bool bound_check(u64 x) const { return (0 < x && x < buffer.size()); }

    GPU_LINE(__host__) u64 to_d1(u64 x, u64 y, u64 z) const { return x + y * (WIDTH) + z * (WIDTH * HEIGHT); }
    GPU_LINE(__host__) u64 to_d1(u64 x, u64 y) const { return x + y * (WIDTH); }

public:
    GPU_LINE(__host__)
    Multi_Dimension_View_Array() : WIDTH(0), HEIGHT(0), DEPTH(0) {}

    GPU_LINE(__host__)
    void set_sizes(i64 width, i64 height = 1, i64 depth = 1)
    {
        if (width <= 0) FATAL_ERROR("WIDTH set with incorrect value");
        if (height <= 0) FATAL_ERROR("HEIGHT set with incorrect value");
        if (depth <= 0) FATAL_ERROR("DEPTH set with incorrect value");

        WIDTH = width;
        HEIGHT = height;
        DEPTH = depth;

        buffer.resize(WIDTH * HEIGHT * DEPTH);
    }

    GPU_LINE(__host__) u64 get_width() const { return WIDTH; }
    GPU_LINE(__host__) u64 get_height() const { return HEIGHT; }
    GPU_LINE(__host__) u64 get_depth() const { return DEPTH; }
    GPU_LINE(__host__) u64 get_total_number() const { return buffer.size(); }

    GPU_LINE(__host__)
    T* get(u64 x, u64 y, u64 z)
    {
        BOUND_CHECKS_Multi_Dimension_View_Array(if (!bound_check(x, y, z)) FATAL_ERROR("accesing beyond vector bounds");)

            return &buffer[to_d1(x, y, z)];
    }

    GPU_LINE(__host__)
    T* get(u64 x, u64 y)
    {
        BOUND_CHECKS_Multi_Dimension_View_Array(if (!bound_check(x, y)) FATAL_ERROR("accesing beyond vector bounds");)

            return &buffer[to_d1(x, y)];
    }

    GPU_LINE(__host__)
    T* get(u64 x)
    {
        BOUND_CHECKS_Multi_Dimension_View_Array(if (!(x < buffer.size())) FATAL_ERROR("accesing beyond vector bounds");)

            return &buffer[x];
    }

    coords get_coords(u64 index_1d) const
    {
        BOUND_CHECKS_Multi_Dimension_View_Array(if (!(index_1d < buffer.size())) FATAL_ERROR("accesing beyond vector bounds");)

        coords c;
        c.z = index_1d / (WIDTH * HEIGHT);
        c.y = (index_1d - c.z * (WIDTH * HEIGHT)) / WIDTH;
        c.x = index_1d - c.z * (WIDTH * HEIGHT) - c.y * WIDTH;

        return c;
    }

    GPU_LINE(__host__)
    const std::vector<T>& get_vector() const
    {
        return buffer;
    }
};
