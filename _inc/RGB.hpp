#pragma once
#include "parallel_common.h"

class RGB
{
    u8 r;
    u8 g;
    u8 b;

public:
    RGB(u8 _r = 0, u8 _g = 0, u8 _b = 0) : c_init(r), c_init(g), c_init(b) {}
    RGB(const RGB& other) : r(other.r), g(other.g), b(other.b) {}

    u8 get_r() const { return r; }
    u8 get_g() const { return g; }
    u8 get_b() const { return b; }

    void print(const string& var_name) const
    {
        line(var_name);

        varr((int)r);
        varr((int)g);
        var((int)b);
    }

    RGB operator*(double multi) { return {static_cast<u8>(r * multi), static_cast<u8>(g * multi), static_cast<u8>(b * multi)}; }
    RGB& operator*=(double multi)
    {
        r = static_cast<u8>(r * multi);
        g = static_cast<u8>(g * multi);
        b = static_cast<u8>(b * multi);

        return *this;
    }
    RGB& operator+=(const RGB& other)
    {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }
    RGB operator+(const RGB& other) { return RGB(r + other.r, g + other.g, b + other.b); }
    RGB& operator-=(const RGB& other)
    {
        r -= other.r;
        g -= other.g;
        b -= other.b;

        return *this;
    }
    RGB operator-(const RGB& other) { return RGB(r - other.r, g - other.g, b - other.b); }

    static RGB get_color_from_temperature(unit temperature, unit smallest_T, unit largest_T, const RGB& color_min, const RGB& color_max)
    {
        if (largest_T < temperature) return color_max;
        if (temperature < smallest_T) return color_min;

        // Normalize the temperature value between 0 and 1
        float normalized_temp = (temperature - smallest_T) / (largest_T - smallest_T);

        // Interpolate between the two colors
        int r = static_cast<int>(color_min.get_r() + normalized_temp * (color_max.get_r() - color_min.get_r()));
        int g = static_cast<int>(color_min.get_g() + normalized_temp * (color_max.get_g() - color_min.get_g()));
        int b = static_cast<int>(color_min.get_b() + normalized_temp * (color_max.get_b() - color_min.get_b()));

        return RGB(r, g, b);
    }
};
